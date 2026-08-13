"""Compare paired CPU and MPS outputs produced by evaluate_asr.py."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


PairKey = tuple[str, str]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL contains no rows: {path}")
    return rows


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _index_rows(
    rows: list[dict[str, Any]], label: str, expected_device: str
) -> dict[PairKey, dict[str, Any]]:
    indexed: dict[PairKey, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        utterance_id = row.get("id")
        condition = row.get("condition")
        if not isinstance(utterance_id, str) or not utterance_id.strip():
            raise ValueError(f"{label} row {row_number} requires a non-empty string id")
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError(
                f"{label} row {row_number} requires a non-empty string condition"
            )
        if row.get("status") != "completed":
            raise ValueError(
                f"{label} row {utterance_id}/{condition} is not completed"
            )
        if row.get("device") != expected_device:
            raise ValueError(
                f"{label} row {utterance_id}/{condition} has device "
                f"{row.get('device')!r}, expected {expected_device!r}"
            )
        for field in ("hypothesis_raw", "hypothesis_normalized"):
            if not isinstance(row.get(field), str):
                raise ValueError(
                    f"{label} row {utterance_id}/{condition} requires string {field}"
                )
        key = (utterance_id.strip(), condition.strip())
        if key in indexed:
            raise ValueError(f"duplicate {label} pair: {key}")
        indexed[key] = row
    if not indexed:
        raise ValueError(f"{label} contains no rows")
    return indexed


def _common_summary(rows: dict[PairKey, dict[str, Any]], label: str) -> dict[str, str]:
    summary: dict[str, str] = {}
    for field in ("model_sha256", "asr_config_digest", "evaluator_code_sha256"):
        values = {row.get(field) for row in rows.values()}
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"all {label} rows require non-empty {field}")
        if len(values) != 1:
            raise ValueError(f"{label} rows contain inconsistent {field} values")
        summary[field] = str(next(iter(values)))
    return summary


def _rtf_summary(rows: dict[PairKey, dict[str, Any]], label: str) -> dict[str, Any]:
    asr_seconds = 0.0
    audio_seconds = 0.0
    per_row_rtf: list[float] = []
    for key, row in rows.items():
        try:
            row_asr_seconds = float(row["asr_seconds"])
            duration_seconds = float(row["duration_seconds"])
            row_rtf = float(row["asr_rtf"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{label} row {key} has invalid ASR timing fields") from error
        if not math.isfinite(row_asr_seconds) or row_asr_seconds < 0:
            raise ValueError(f"{label} row {key} has invalid asr_seconds")
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError(f"{label} row {key} has invalid duration_seconds")
        if not math.isfinite(row_rtf) or row_rtf < 0:
            raise ValueError(f"{label} row {key} has invalid asr_rtf")
        if not math.isclose(
            row_rtf,
            row_asr_seconds / duration_seconds,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{label} row {key} has inconsistent ASR timing fields")
        asr_seconds += row_asr_seconds
        audio_seconds += duration_seconds
        per_row_rtf.append(row_rtf)
    return {
        "asr_seconds": asr_seconds,
        "audio_seconds": audio_seconds,
        "asr_rtf": asr_seconds / audio_seconds,
        "mean_utterance_asr_rtf": sum(per_row_rtf) / len(per_row_rtf),
    }


def compare_devices(
    cpu_rows: list[dict[str, Any]], mps_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Strictly pair device outputs and return an equivalence report."""

    cpu = _index_rows(cpu_rows, "CPU", "cpu")
    mps = _index_rows(mps_rows, "MPS", "mps")
    cpu_keys = set(cpu)
    mps_keys = set(mps)
    if cpu_keys != mps_keys:
        missing_from_mps = sorted(cpu_keys - mps_keys)
        missing_from_cpu = sorted(mps_keys - cpu_keys)
        raise ValueError(
            "CPU/MPS pair sets differ: "
            f"missing_from_mps={missing_from_mps}, missing_from_cpu={missing_from_cpu}"
        )

    cpu_summary = _common_summary(cpu, "CPU")
    mps_summary = _common_summary(mps, "MPS")
    if cpu_summary != mps_summary:
        raise ValueError(
            f"CPU/MPS runtime identity differs: {cpu_summary} != {mps_summary}"
        )

    raw_exact_matches = 0
    normalized_exact_matches = 0
    mismatches: list[dict[str, Any]] = []
    for utterance_id, condition in sorted(cpu):
        cpu_row = cpu[(utterance_id, condition)]
        mps_row = mps[(utterance_id, condition)]
        for identity_field in (
            "audio_sha256",
            "reference_raw_sha256",
            "num_samples",
            "duration_seconds",
        ):
            if cpu_row.get(identity_field) != mps_row.get(identity_field):
                raise ValueError(
                    f"CPU/MPS pair {utterance_id}/{condition} differs in "
                    f"{identity_field}"
                )
        raw_match = cpu_row["hypothesis_raw"] == mps_row["hypothesis_raw"]
        normalized_match = (
            cpu_row["hypothesis_normalized"]
            == mps_row["hypothesis_normalized"]
        )
        raw_exact_matches += int(raw_match)
        normalized_exact_matches += int(normalized_match)
        if not raw_match or not normalized_match:
            mismatches.append(
                {
                    "id": utterance_id,
                    "condition": condition,
                    "raw_match": raw_match,
                    "normalized_match": normalized_match,
                    "cpu": {
                        "hypothesis_raw": cpu_row["hypothesis_raw"],
                        "hypothesis_normalized": cpu_row["hypothesis_normalized"],
                    },
                    "mps": {
                        "hypothesis_raw": mps_row["hypothesis_raw"],
                        "hypothesis_normalized": mps_row[
                            "hypothesis_normalized"
                        ],
                    },
                }
            )

    row_count = len(cpu)
    normalized_all_exact = normalized_exact_matches == row_count
    return {
        "schema_version": 1,
        "status": "passed" if normalized_all_exact else "failed",
        "pairing_key": ["id", "condition"],
        "rows_compared": row_count,
        **cpu_summary,
        "exact_matches": {
            "raw": raw_exact_matches,
            "normalized": normalized_exact_matches,
        },
        "all_exact": {
            "raw": raw_exact_matches == row_count,
            "normalized": normalized_all_exact,
        },
        "mismatch_count": len(mismatches),
        "normalized_mismatch_count": row_count - normalized_exact_matches,
        "mismatches": mismatches,
        "rtf": {
            "cpu": _rtf_summary(cpu, "CPU"),
            "mps": _rtf_summary(mps, "MPS"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True, help="CPU hypothesis JSONL")
    parser.add_argument("--mps", type=Path, required=True, help="MPS hypothesis JSONL")
    parser.add_argument(
        "--output", type=Path, default=Path("device_equivalence.json")
    )
    arguments = parser.parse_args()

    try:
        report = compare_devices(read_jsonl(arguments.cpu), read_jsonl(arguments.mps))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        report = {
            "schema_version": 1,
            "status": "failed",
            "validation_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        atomic_write_json(arguments.output, report)
        return 2
    atomic_write_json(arguments.output, report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
