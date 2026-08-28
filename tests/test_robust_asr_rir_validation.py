from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robust_asr.acoustics import bank
from robust_asr.acoustics.bank import generate_rir_bank
from robust_asr.acoustics.pyroom import SimulatedRIR
from robust_asr.acoustics.validation import validate_formal_rir_banks


def _fake_simulation(scene, *, target_rt60_seconds, seed):
    full = np.zeros((4, 8), dtype=np.float32)
    full[:, 0] = 1.0
    return SimulatedRIR(
        full=full,
        direct=full[:, :1].copy(),
        target_rt60_seconds=float(target_rt60_seconds),
        measured_rt60_seconds=(target_rt60_seconds,) * 4,
        design_rt60_seconds=float(target_rt60_seconds),
        absorption=0.5,
        max_order=1,
        drr_db=(10.0,) * 4,
        seed=seed,
        calibration_iterations=1,
    )


def _banks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bank, "simulate_calibrated_rir", _fake_simulation)
    generate_rir_bank(
        tmp_path,
        split="train",
        rooms=1,
        train_positions_per_room=2,
    )
    generate_rir_bank(
        tmp_path,
        split="dev",
        rooms=1,
        positions_per_target=1,
        fixed_rt60_seconds=(0.2, 0.8),
    )
    generate_rir_bank(
        tmp_path,
        split="test",
        rooms=1,
        positions_per_target=1,
        fixed_rt60_seconds=(0.2, 0.8),
    )


def _validate(tmp_path: Path, *, workers: int = 1):
    return validate_formal_rir_banks(
        tmp_path,
        train_rooms=1,
        train_positions_per_room=2,
        dev_rooms=1,
        dev_positions_per_rt60=1,
        test_rooms=1,
        test_positions_per_rt60=1,
        fixed_rt60_seconds=(0.2, 0.8),
        workers=workers,
    )


def test_validate_formal_rir_banks_checks_inventory_and_pairing(
    tmp_path: Path, monkeypatch
) -> None:
    _banks(tmp_path, monkeypatch)

    summary = _validate(tmp_path)

    assert summary["status"] == "PASS"
    assert summary["splits"]["train"]["rirs"] == 2
    assert summary["splits"]["test"]["paired_rt60_geometry"] is True


def test_validate_formal_rir_banks_rejects_corrupted_file(
    tmp_path: Path, monkeypatch
) -> None:
    _banks(tmp_path, monkeypatch)
    target = next((tmp_path / "test").glob("*.npz"))
    target.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validate(tmp_path)


def test_validate_formal_rir_banks_parallel_file_checks(
    tmp_path: Path, monkeypatch
) -> None:
    _banks(tmp_path, monkeypatch)

    summary = _validate(tmp_path, workers=4)

    assert summary["status"] == "PASS"
    assert summary["workers"] == 4


def test_validate_formal_rir_banks_rejects_invalid_worker_count(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        _validate(tmp_path, workers=0)


def test_validate_formal_rir_banks_supports_legacy_dev_audit(
    tmp_path: Path, monkeypatch
) -> None:
    _banks(tmp_path, monkeypatch)
    audit_path = tmp_path / "dev.audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.pop("geometry_families")
    audit.pop("paired_rt60_geometry")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    summary = _validate(tmp_path, workers=2)

    assert summary["splits"]["dev"]["geometry_families"] == 2
    assert summary["splits"]["dev"]["paired_rt60_geometry"] is False
