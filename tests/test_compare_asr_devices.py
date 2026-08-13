from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_asr_devices.py"
SPEC = importlib.util.spec_from_file_location("compare_asr_devices", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(
    utterance_id: str,
    condition: str,
    raw: str,
    normalized: str,
    *,
    device: str = "cpu",
    asr_seconds: float = 0.5,
    duration_seconds: float = 1.0,
) -> dict[str, object]:
    return {
        "id": utterance_id,
        "condition": condition,
        "status": "completed",
        "model_sha256": "f" * 64,
        "asr_config_digest": "e" * 64,
        "evaluator_code_sha256": "d" * 64,
        "device": device,
        "audio_sha256": (utterance_id + condition).ljust(64, "0")[:64],
        "reference_raw_sha256": utterance_id.ljust(64, "0")[:64],
        "num_samples": int(duration_seconds * 16_000),
        "hypothesis_raw": raw,
        "hypothesis_normalized": normalized,
        "asr_seconds": asr_seconds,
        "duration_seconds": duration_seconds,
        "asr_rtf": asr_seconds / duration_seconds,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_report_pairs_by_key_and_aggregates_corpus_rtf() -> None:
    cpu = [
        _row("u2", "noisy", "Goodbye.", "goodbye", asr_seconds=2.0),
        _row("u1", "clean", "Hello!", "hello", asr_seconds=0.5),
    ]
    mps = [
        _row("u1", "clean", "Hello", "hello", device="mps", asr_seconds=0.2),
        _row("u2", "noisy", "Goodbye.", "goodbye", device="mps", asr_seconds=0.8),
    ]

    report = MODULE.compare_devices(cpu, mps)

    assert report["status"] == "passed"
    assert report["exact_matches"] == {"raw": 1, "normalized": 2}
    assert report["all_exact"] == {"raw": False, "normalized": True}
    assert report["mismatch_count"] == 1
    assert report["mismatches"][0]["id"] == "u1"
    assert report["rtf"]["cpu"]["asr_rtf"] == 1.25
    assert report["rtf"]["mps"]["asr_rtf"] == 0.5


def test_normalized_mismatch_writes_failed_report_and_exits_nonzero(
    tmp_path: Path,
) -> None:
    cpu_path = tmp_path / "cpu.jsonl"
    mps_path = tmp_path / "mps.jsonl"
    output = tmp_path / "device_equivalence.json"
    _write_jsonl(cpu_path, [_row("u1", "noisy", "cat", "cat")])
    _write_jsonl(
        mps_path, [_row("u1", "noisy", "cap", "cap", device="mps")]
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cpu",
            str(cpu_path),
            "--mps",
            str(mps_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["normalized_mismatch_count"] == 1
    assert report["mismatches"][0]["normalized_match"] is False


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows.append(dict(rows[0])), "duplicate CPU pair"),
        (lambda rows: rows.pop(), "pair sets differ"),
        (
            lambda rows: [
                row.__setitem__("model_sha256", "0" * 64) for row in rows
            ],
            "runtime identity differs",
        ),
        (
            lambda rows: [
                row.__setitem__("asr_config_digest", "0" * 64) for row in rows
            ],
            "runtime identity differs",
        ),
        (
            lambda rows: [
                row.__setitem__("evaluator_code_sha256", "0" * 64) for row in rows
            ],
            "runtime identity differs",
        ),
    ],
)
def test_strict_pairing_and_summary_validation(change: object, message: str) -> None:
    cpu = [
        _row("u1", "clean", "one", "one"),
        _row("u2", "noisy", "two", "two"),
    ]
    mps = [
        _row("u1", "clean", "one", "one", device="mps"),
        _row("u2", "noisy", "two", "two", device="mps"),
    ]
    change(cpu)

    with pytest.raises(ValueError, match=message):
        MODULE.compare_devices(cpu, mps)


def test_inconsistent_row_rtf_is_rejected() -> None:
    cpu = [_row("u1", "clean", "one", "one")]
    mps = [_row("u1", "clean", "one", "one", device="mps")]
    cpu[0]["asr_rtf"] = 999.0

    with pytest.raises(ValueError, match="inconsistent ASR timing"):
        MODULE.compare_devices(cpu, mps)


def test_cli_structural_failure_replaces_old_report(tmp_path: Path) -> None:
    cpu_path = tmp_path / "cpu.jsonl"
    mps_path = tmp_path / "mps.jsonl"
    output = tmp_path / "device_equivalence.json"
    _write_jsonl(cpu_path, [_row("u1", "clean", "one", "one")])
    _write_jsonl(mps_path, [_row("u1", "clean", "one", "one")])
    output.write_text('{"status":"old"}\n', encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cpu",
            str(cpu_path),
            "--mps",
            str(mps_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "expected 'mps'" in report["validation_error"]["message"]
