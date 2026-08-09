"""Audit the prepared 20/8/official-test VoiceBank evaluation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def partition_summary(
    rows: list[dict[str, Any]], project_root: Path
) -> dict[str, Any]:
    ids = [str(row["id"]) for row in rows]
    speakers = sorted({str(row["speaker_id"]) for row in rows})
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate utterance id within partition")
    bad_rates = sorted(
        {int(row["sample_rate"]) for row in rows if int(row["sample_rate"]) != 16_000}
    )
    if bad_rates:
        raise ValueError(f"unexpected sample rates: {bad_rates}")
    missing = []
    for row in rows:
        for key in ("clean", "noisy"):
            path = Path(str(row[key]))
            resolved = path if path.is_absolute() else project_root / path
            if not resolved.is_file():
                missing.append(resolved.as_posix())
    if missing:
        raise FileNotFoundError(f"missing prepared audio: {missing[:5]}")
    total_samples = sum(int(row["num_samples"]) for row in rows)
    return {
        "pairs": len(rows),
        "speakers": len(speakers),
        "speaker_ids": speakers,
        "duration_hours": total_samples / 16_000 / 3_600,
        "sample_rate_hz": 16_000,
    }


def audit_protocol(
    partitions: dict[str, list[dict[str, Any]]],
    project_root: Path,
    *,
    expected_speakers: dict[str, int],
) -> dict[str, Any]:
    report = {
        name: partition_summary(rows, project_root)
        for name, rows in partitions.items()
    }
    for name, expected in expected_speakers.items():
        actual = int(report[name]["speakers"])
        if actual != expected:
            raise ValueError(
                f"{name} speaker count mismatch: expected {expected}, got {actual}"
            )

    names = list(partitions)
    for index, first_name in enumerate(names):
        first_ids = {str(row["id"]) for row in partitions[first_name]}
        first_speakers = {
            str(row["speaker_id"]) for row in partitions[first_name]
        }
        for second_name in names[index + 1 :]:
            second_ids = {str(row["id"]) for row in partitions[second_name]}
            second_speakers = {
                str(row["speaker_id"]) for row in partitions[second_name]
            }
            if first_ids & second_ids:
                raise ValueError(
                    f"utterance leakage between {first_name} and {second_name}"
                )
            if first_speakers & second_speakers:
                raise ValueError(
                    f"speaker leakage between {first_name} and {second_name}"
                )
    return {
        "status": "passed",
        "partitions": report,
        "total_pairs": sum(len(rows) for rows in partitions.values()),
        "speaker_disjoint": True,
        "utterance_disjoint": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("data/manifests")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/full_protocol/data_audit.json"),
    )
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()
    manifest_root = (
        arguments.manifest_root
        if arguments.manifest_root.is_absolute()
        else project_root / arguments.manifest_root
    )
    partitions = {
        name: read_jsonl(manifest_root / f"{name}.jsonl")
        for name in ("development", "validation", "official_test")
    }
    report = audit_protocol(
        partitions,
        project_root,
        expected_speakers={
            "development": 20,
            "validation": 8,
            "official_test": 2,
        },
    )
    output = (
        arguments.output
        if arguments.output.is_absolute()
        else project_root / arguments.output
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
