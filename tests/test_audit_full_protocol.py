from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_full_protocol.py"
SPEC = importlib.util.spec_from_file_location("audit_full_protocol", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_row(root: Path, speaker: str, index: int) -> dict[str, object]:
    clean = root / f"{speaker}_{index}_clean.wav"
    noisy = root / f"{speaker}_{index}_noisy.wav"
    clean.touch()
    noisy.touch()
    return {
        "id": f"{speaker}_{index}",
        "speaker_id": speaker,
        "clean": clean.relative_to(root).as_posix(),
        "noisy": noisy.relative_to(root).as_posix(),
        "sample_rate": 16_000,
        "num_samples": 16_000,
    }


def test_audit_protocol_accepts_disjoint_partitions(tmp_path: Path) -> None:
    partitions = {
        "development": [make_row(tmp_path, "p1", 1)],
        "validation": [make_row(tmp_path, "p2", 1)],
        "official_test": [make_row(tmp_path, "p3", 1)],
    }
    report = MODULE.audit_protocol(
        partitions,
        tmp_path,
        expected_speakers={
            "development": 1,
            "validation": 1,
            "official_test": 1,
        },
    )
    assert report["status"] == "passed"
    assert report["total_pairs"] == 3


def test_audit_protocol_rejects_speaker_leakage(tmp_path: Path) -> None:
    partitions = {
        "development": [make_row(tmp_path, "p1", 1)],
        "validation": [make_row(tmp_path, "p1", 2)],
    }
    with pytest.raises(ValueError, match="speaker leakage"):
        MODULE.audit_protocol(
            partitions,
            tmp_path,
            expected_speakers={"development": 1, "validation": 1},
        )
