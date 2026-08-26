from __future__ import annotations

import numpy as np
import pytest

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
