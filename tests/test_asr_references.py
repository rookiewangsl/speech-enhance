from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_asr_references.py"
SPEC = importlib.util.spec_from_file_location("prepare_asr_references", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def make_manifest(path: Path) -> None:
    write_jsonl(
        path,
        [
            manifest_row("p227_001", "development"),
            manifest_row("p232_001", "official_test"),
        ],
    )


def manifest_row(utterance_id: str, split: str) -> dict[str, object]:
    speaker = utterance_id.split("_", maxsplit=1)[0]
    return {
        "id": utterance_id,
        "speaker_id": speaker,
        "split": split,
        "clean": f"clean/{utterance_id}.wav",
        "noisy": f"noisy/{utterance_id}.wav",
    }


def make_transcript(root: Path, utterance_id: str, text: str) -> Path:
    speaker = utterance_id.split("_", maxsplit=1)[0]
    path = root / speaker / f"{utterance_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_prepare_references_writes_strict_mapping_and_audit(tmp_path: Path) -> None:
    manifest = tmp_path / "development.jsonl"
    transcripts = tmp_path / "txt"
    make_manifest(manifest)
    make_transcript(transcripts, "p227_001", "Please call Stella.\n")
    make_transcript(transcripts, "p232_001", "A second sentence.\n")
    make_transcript(transcripts, "p999_001", "Unused source transcript.\n")
    make_transcript(transcripts, "._p227_001", "macOS metadata sidecar")
    output = tmp_path / "references.jsonl"
    audit_output = tmp_path / "audit.json"

    report = MODULE.prepare_asr_references(
        [manifest], transcripts, output, audit_output
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [
        {
            "id": "p227_001",
            "reference_raw": "Please call Stella.",
            "speaker_id": "p227",
            "split": "development",
            "transcript_relative_path": "p227/p227_001.txt",
            "transcript_source": "VCTK",
            "transcript_version": "0.92",
        },
        {
            "id": "p232_001",
            "reference_raw": "A second sentence.",
            "speaker_id": "p232",
            "split": "official_test",
            "transcript_relative_path": "p232/p232_001.txt",
            "transcript_source": "VCTK",
            "transcript_version": "0.92",
        },
    ]
    audit = json.loads(audit_output.read_text())
    assert report == audit
    assert audit["status"] == "passed"
    assert audit["counts"] == {
        "manifest_rows": 2,
        "reference_rows": 2,
        "transcript_files": 3,
        "consumed_transcript_files": 2,
        "unused_transcript_files": 1,
    }
    assert audit["checks"] == {
        "duplicate_manifest_ids": [],
        "duplicate_transcript_ids": [],
        "empty_transcripts": [],
        "invalid_manifest_rows": [],
        "missing_transcripts": [],
        "speaker_mismatches": [],
    }
    assert audit["output"]["sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert audit["manual_review"]["status"] == "pending"
    assert {row["id"] for row in audit["manual_review"]["candidates"]} == {
        "p227_001",
        "p232_001",
    }
    assert not list(tmp_path.rglob("*.part"))


def test_rejects_duplicate_manifest_ids_and_preserves_old_output(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    row = manifest_row("p227_001", "development")
    write_jsonl(first, [row])
    write_jsonl(second, [{**row, "split": "validation"}])
    transcripts = tmp_path / "txt"
    make_transcript(transcripts, "p227_001", "Some words.")
    output = tmp_path / "references.jsonl"
    output.write_text("old result\n", encoding="utf-8")
    audit_output = tmp_path / "audit.json"

    with pytest.raises(MODULE.ReferenceAuditError, match="duplicate_manifest_ids"):
        MODULE.prepare_asr_references(
            [first, second], transcripts, output, audit_output
        )

    assert output.read_text(encoding="utf-8") == "old result\n"
    audit = json.loads(audit_output.read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert audit["checks"]["duplicate_manifest_ids"] == ["p227_001"]


@pytest.mark.parametrize(
    ("problem", "expected_check"),
    [
        ("missing", "missing_transcripts"),
        ("empty", "empty_transcripts"),
        ("speaker", "speaker_mismatches"),
    ],
)
def test_rejects_missing_empty_and_speaker_mismatch(
    tmp_path: Path, problem: str, expected_check: str
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [manifest_row("p227_001", "development")],
    )
    transcripts = tmp_path / "txt"
    transcripts.mkdir()
    if problem == "empty":
        make_transcript(transcripts, "p227_001", "  \n")
    elif problem == "speaker":
        wrong = transcripts / "p999" / "p227_001.txt"
        wrong.parent.mkdir()
        wrong.write_text("Some words.", encoding="utf-8")

    with pytest.raises(MODULE.ReferenceAuditError) as captured:
        MODULE.build_reference_records(
            [manifest],
            transcripts,
            transcript_source="VCTK",
            transcript_version="0.92",
        )

    assert captured.value.report["checks"][expected_check]


def test_rejects_duplicate_transcript_stems(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [manifest_row("p227_001", "development")],
    )
    transcripts = tmp_path / "txt"
    make_transcript(transcripts, "p227_001", "First copy.")
    duplicate = transcripts / "duplicate" / "p227_001.txt"
    duplicate.parent.mkdir()
    duplicate.write_text("Second copy.", encoding="utf-8")

    with pytest.raises(
        MODULE.ReferenceAuditError, match="duplicate_transcript_ids"
    ):
        MODULE.build_reference_records(
            [manifest],
            transcripts,
            transcript_source="VCTK",
            transcript_version="0.92",
        )
