"""Map VoiceBank manifests to official VCTK transcripts with strict auditing."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


class ReferenceAuditError(ValueError):
    """Raised when a reference audit fails.

    ``report`` is safe to serialize and is also written by the command-line
    entry point, so a failed preparation remains diagnosable without producing
    a partial reference manifest.
    """

    def __init__(self, report: dict[str, Any]):
        self.report = report
        failures = [
            name for name, values in report["checks"].items() if values
        ]
        super().__init__("ASR reference audit failed: " + ", ".join(failures))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL manifest and report malformed rows with their line number."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"manifest row must be an object: {path}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _speaker_from_utterance_id(utterance_id: str) -> str | None:
    speaker, separator, utterance = utterance_id.partition("_")
    if not speaker or not separator or not utterance:
        return None
    return speaker


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _jsonl_text(records: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically write a human-readable JSON object."""

    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def build_reference_records(
    manifest_paths: list[Path],
    transcript_root: Path,
    *,
    transcript_source: str,
    transcript_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build reference rows and a complete pass/fail audit report.

    VCTK files are located by utterance ID, but a match is accepted only when
    its immediate parent directory equals the manifest speaker ID.  All
    failures are collected before the function rejects the source data.
    """

    if not manifest_paths:
        raise ValueError("at least one manifest is required")
    if not transcript_source.strip():
        raise ValueError("transcript_source must not be empty")
    if not transcript_version.strip():
        raise ValueError("transcript_version must not be empty")
    if not transcript_root.is_dir():
        raise FileNotFoundError(
            f"VCTK transcript root is not a directory: {transcript_root}"
        )

    manifest_entries: list[tuple[Path, int, dict[str, Any]]] = []
    for manifest_path in manifest_paths:
        for line_number, row in enumerate(read_jsonl(manifest_path), start=1):
            manifest_entries.append((manifest_path, line_number, row))

    transcript_paths = sorted(
        path
        for path in transcript_root.rglob("*.txt")
        if path.is_file() and not path.name.startswith("._")
    )
    paths_by_id: dict[str, list[Path]] = {}
    for path in transcript_paths:
        paths_by_id.setdefault(path.stem, []).append(path)

    id_counts = Counter(str(row.get("id", "")) for _, _, row in manifest_entries)
    duplicate_manifest_ids = sorted(
        utterance_id
        for utterance_id, count in id_counts.items()
        if utterance_id and count > 1
    )
    duplicate_transcript_ids = sorted(
        utterance_id
        for utterance_id in id_counts
        if utterance_id and len(paths_by_id.get(utterance_id, [])) > 1
    )

    missing_transcripts: list[str] = []
    empty_transcripts: list[str] = []
    invalid_manifest_rows: list[str] = []
    speaker_mismatches: list[str] = []
    records: list[dict[str, Any]] = []
    consumed_transcripts: set[Path] = set()

    for manifest_path, line_number, row in manifest_entries:
        location = f"{manifest_path.as_posix()}:{line_number}"
        utterance_id = row.get("id")
        speaker_id = row.get("speaker_id")
        split = row.get("split")
        clean_path = row.get("clean")
        noisy_path = row.get("noisy")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (utterance_id, speaker_id, split, clean_path, noisy_path)
        ):
            invalid_manifest_rows.append(location)
            continue
        assert isinstance(utterance_id, str)
        assert isinstance(speaker_id, str)
        assert isinstance(split, str)
        assert isinstance(clean_path, str)
        assert isinstance(noisy_path, str)
        if Path(clean_path).stem != utterance_id or Path(noisy_path).stem != utterance_id:
            invalid_manifest_rows.append(
                f"{location}: clean/noisy stem must equal {utterance_id}"
            )
            continue
        derived_speaker = _speaker_from_utterance_id(utterance_id)
        if derived_speaker != speaker_id:
            speaker_mismatches.append(
                f"{utterance_id}: manifest speaker={speaker_id}, "
                f"id speaker={derived_speaker}"
            )

        candidates = paths_by_id.get(utterance_id, [])
        if not candidates:
            missing_transcripts.append(utterance_id)
            continue
        if len(candidates) != 1:
            continue
        transcript_path = candidates[0]
        if transcript_path.parent.name != speaker_id:
            speaker_mismatches.append(
                f"{utterance_id}: transcript directory="
                f"{transcript_path.parent.name}, manifest speaker={speaker_id}"
            )
            continue
        raw_text = transcript_path.read_text(encoding="utf-8-sig").strip()
        if not raw_text:
            empty_transcripts.append(utterance_id)
            continue
        consumed_transcripts.add(transcript_path)
        records.append(
            {
                "id": utterance_id,
                "speaker_id": speaker_id,
                "split": split,
                "reference_raw": raw_text,
                "transcript_source": transcript_source,
                "transcript_version": transcript_version,
                "transcript_relative_path": _relative_path(
                    transcript_path, transcript_root
                ),
            }
        )

    checks: dict[str, list[str]] = {
        "duplicate_manifest_ids": duplicate_manifest_ids,
        "duplicate_transcript_ids": duplicate_transcript_ids,
        "missing_transcripts": sorted(set(missing_transcripts)),
        "empty_transcripts": sorted(set(empty_transcripts)),
        "invalid_manifest_rows": sorted(set(invalid_manifest_rows)),
        "speaker_mismatches": sorted(set(speaker_mismatches)),
    }
    partition_speakers: dict[str, set[str]] = {}
    partition_rows: Counter[str] = Counter()
    for record in records:
        partition = str(record["split"])
        partition_rows[partition] += 1
        partition_speakers.setdefault(partition, set()).add(
            str(record["speaker_id"])
        )
    manifest_by_id = {
        str(row.get("id")): row for _, _, row in manifest_entries if row.get("id")
    }
    records_by_partition: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_partition.setdefault(str(record["split"]), []).append(record)
    manual_candidates: list[dict[str, Any]] = []
    for partition, partition_records in sorted(records_by_partition.items()):
        ordered = sorted(partition_records, key=lambda record: str(record["id"]))
        indices = sorted({0, len(ordered) // 2, len(ordered) - 1})
        for index in indices:
            record = ordered[index]
            source_row = manifest_by_id[str(record["id"])]
            manual_candidates.append(
                {
                    "id": record["id"],
                    "speaker_id": record["speaker_id"],
                    "split": partition,
                    "clean_audio": source_row["clean"],
                    "transcript_relative_path": record["transcript_relative_path"],
                }
            )

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed" if any(checks.values()) else "passed",
        "transcript_source": transcript_source,
        "transcript_version": transcript_version,
        "transcript_root": transcript_root.resolve().as_posix(),
        "manifests": [
            {
                "path": path.resolve().as_posix(),
                "sha256": _sha256(path),
            }
            for path in manifest_paths
        ],
        "counts": {
            "manifest_rows": len(manifest_entries),
            "reference_rows": len(records),
            "transcript_files": len(transcript_paths),
            "consumed_transcript_files": len(consumed_transcripts),
            "unused_transcript_files": len(transcript_paths)
            - len(consumed_transcripts),
        },
        "partitions": {
            partition: {
                "utterances": partition_rows[partition],
                "speakers": len(partition_speakers[partition]),
                "speaker_ids": sorted(partition_speakers[partition]),
            }
            for partition in sorted(partition_rows)
        },
        "checks": checks,
        "manual_review": {
            "status": "pending",
            "purpose": "Listen to clean audio while reading the official transcript; do not use ASR output as reference.",
            "candidates": manual_candidates,
        },
    }
    if report["status"] == "failed":
        raise ReferenceAuditError(report)
    return records, report


def prepare_asr_references(
    manifest_paths: list[Path],
    transcript_root: Path,
    output_path: Path,
    audit_output_path: Path,
    *,
    transcript_source: str = "VCTK",
    transcript_version: str = "0.92",
) -> dict[str, Any]:
    """Validate inputs and atomically produce a reference JSONL and audit JSON."""

    try:
        records, report = build_reference_records(
            manifest_paths,
            transcript_root,
            transcript_source=transcript_source,
            transcript_version=transcript_version,
        )
    except ReferenceAuditError as error:
        atomic_write_json(audit_output_path, error.report)
        raise

    reference_text = _jsonl_text(records)
    report["output"] = {
        "path": output_path.resolve().as_posix(),
        "sha256": hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
    }
    _atomic_write_text(output_path, reference_text)
    atomic_write_json(audit_output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly map VoiceBank manifest IDs to VCTK transcripts."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        dest="manifests",
        help=(
            "VoiceBank JSONL manifest; repeat for multiple partitions. "
            "Defaults to development, validation, and official_test."
        ),
    )
    parser.add_argument(
        "--transcript-root",
        type=Path,
        required=True,
        help="Path to the VCTK txt directory (containing speaker subdirectories).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/asr_references.jsonl"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("outputs/asr/reference_audit.json"),
    )
    parser.add_argument("--transcript-source", default="VCTK")
    parser.add_argument("--transcript-version", default="0.92")
    arguments = parser.parse_args()
    manifests = arguments.manifests or [
        Path("data/manifests/development.jsonl"),
        Path("data/manifests/validation.jsonl"),
        Path("data/manifests/official_test.jsonl"),
    ]
    report = prepare_asr_references(
        manifests,
        arguments.transcript_root,
        arguments.output,
        arguments.audit_output,
        transcript_source=arguments.transcript_source,
        transcript_version=arguments.transcript_version,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
