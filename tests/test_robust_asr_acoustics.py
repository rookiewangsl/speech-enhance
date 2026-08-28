from __future__ import annotations

import json

import numpy as np
import pytest

from robust_asr.acoustics import bank
from robust_asr.acoustics.bank import RIRBankProgress, generate_rir_bank
from robust_asr.acoustics.geometry import (
    GeometryProtocol,
    sample_scene_geometry,
    uniform_circular_array,
)
from robust_asr.acoustics.rir import (
    convolve_multichannel,
    direct_to_reverberant_ratio,
    rt60_within_tolerance,
)
from robust_asr.manifest import read_jsonl


def test_scene_geometry_is_deterministic_and_valid() -> None:
    protocol = GeometryProtocol()
    first = sample_scene_geometry(2026, protocol)
    second = sample_scene_geometry(2026, protocol)

    assert first == second
    assert len(first.microphone_positions_m) == 4
    assert protocol.source_distance.minimum <= first.source_array_distance_m
    assert first.source_array_distance_m <= protocol.source_distance.maximum
    room = np.asarray(first.room_dimensions_m)
    source = np.asarray(first.source_position_m)
    assert protocol.wall_margin_m <= source[0] <= room[0] - protocol.wall_margin_m
    assert protocol.wall_margin_m <= source[1] <= room[1] - protocol.wall_margin_m


def test_uniform_circular_array_has_requested_radius() -> None:
    center = np.array([2.0, 3.0, 0.8])
    positions = uniform_circular_array(tuple(center), radius_m=0.05)
    distances = np.linalg.norm(positions[:, :2] - center[:2], axis=1)

    np.testing.assert_allclose(distances, 0.05)
    np.testing.assert_allclose(positions[:, 2], 0.8)


def test_multichannel_convolution_preserves_tail_and_common_scale() -> None:
    clean = np.array([1.0, 0.5, -0.25])
    rirs = np.array(
        [
            [1.0, 0.5, 0.0, 0.1],
            [0.5, 0.25, 0.0, 0.05],
            [0.8, 0.4, 0.0, 0.08],
            [0.6, 0.3, 0.0, 0.06],
        ]
    )

    result = convolve_multichannel(clean, rirs)

    assert result.signals.shape == (4, len(clean) + rirs.shape[1] - 1)
    np.testing.assert_allclose(result.signals[1], result.signals[0] * 0.5)
    assert np.max(np.abs(result.signals)) <= 10 ** (-1 / 20) + 1e-12


def test_oracle_drr_uses_direct_and_residual_energy() -> None:
    direct = np.array([[1.0, 0.0], [2.0, 0.0]])
    full = np.array([[1.0, 1.0], [2.0, 1.0]])

    drr = direct_to_reverberant_ratio(full, direct)

    np.testing.assert_allclose(drr, [0.0, 10 * np.log10(4.0)])


def test_rt60_tolerance_uses_max_absolute_or_relative_rule() -> None:
    assert rt60_within_tolerance(0.24, 0.2)
    assert not rt60_within_tolerance(0.26, 0.2)
    assert rt60_within_tolerance(1.09, 1.0)
    assert not rt60_within_tolerance(1.11, 1.0)


def test_pyroom_randomized_ism_is_reproducible_when_installed() -> None:
    pytest.importorskip("pyroomacoustics")
    from robust_asr.acoustics.pyroom import simulate_calibrated_rir

    scene = sample_scene_geometry(71)
    first = simulate_calibrated_rir(
        scene, target_rt60_seconds=0.2, seed=99
    )
    second = simulate_calibrated_rir(
        scene, target_rt60_seconds=0.2, seed=99
    )

    assert np.array_equal(first.full, second.full)
    assert first.metadata() == second.metadata()


def _fake_simulation(scene, *, target_rt60_seconds, seed):
    from robust_asr.acoustics.pyroom import SimulatedRIR

    full = np.zeros((4, 8), dtype=np.float32)
    full[:, 0] = 1.0
    direct = full[:, :1].copy()
    return SimulatedRIR(
        full=full,
        direct=direct,
        target_rt60_seconds=float(target_rt60_seconds),
        measured_rt60_seconds=(target_rt60_seconds,) * 4,
        design_rt60_seconds=float(target_rt60_seconds),
        absorption=0.5,
        max_order=1,
        drr_db=(10.0,) * 4,
        seed=seed,
        calibration_iterations=1,
    )


def test_formal_test_rir_families_pair_geometry_across_rt60(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(bank, "simulate_calibrated_rir", _fake_simulation)

    audit = generate_rir_bank(
        tmp_path,
        split="test",
        rooms=1,
        positions_per_target=2,
        fixed_rt60_seconds=(0.2, 0.8),
    )
    rows = read_jsonl(tmp_path / "test.jsonl")

    assert audit["rirs"] == 4
    assert audit["geometry_families"] == 2
    assert audit["paired_rt60_geometry"] is True
    families = {row["rir_family_id"] for row in rows}
    for family in families:
        values = [row for row in rows if row["rir_family_id"] == family]
        assert {row["target_rt60_seconds"] for row in values} == {0.2, 0.8}
        assert len({row["geometry_id"] for row in values}) == 1
    assert len({json.dumps(row["scene"], sort_keys=True) for row in values}) == 1


def test_rir_bank_reports_deterministic_progress(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bank, "simulate_calibrated_rir", _fake_simulation)
    events: list[RIRBankProgress] = []

    generate_rir_bank(
        tmp_path,
        split="test",
        rooms=1,
        positions_per_target=2,
        fixed_rt60_seconds=(0.2, 0.8),
        progress_callback=events.append,
    )

    assert events[0].completed == 0
    assert events[0].total == 4
    assert [event.completed for event in events[1:]] == [1, 2, 3, 4]
    assert events[-1].rir_id == "test_r000_f001_t001"


def test_dev_rir_geometry_remains_independent_per_rt60(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bank, "simulate_calibrated_rir", _fake_simulation)

    audit = generate_rir_bank(
        tmp_path,
        split="dev",
        rooms=1,
        positions_per_target=2,
        fixed_rt60_seconds=(0.2, 0.8),
    )
    rows = read_jsonl(tmp_path / "dev.jsonl")

    assert audit["rirs"] == 4
    assert audit["geometry_families"] == 4
    assert audit["paired_rt60_geometry"] is False
    assert len({row["geometry_id"] for row in rows}) == 4
    assert len({json.dumps(row["scene"], sort_keys=True) for row in rows}) == 4
