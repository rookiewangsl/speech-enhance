"""AISHELL-1 extraction, manifest construction, and split auditing."""

from __future__ import annotations

import re
import tarfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import soundfile as sf
import numpy as np

from .config import canonical_sha256
from .data import (
    CleanUtterance,
    select_speaker_balanced_count_subset,
    select_speaker_balanced_duration_subset,
)
from .manifest import validate_disjoint_groups, write_jsonl_atomic
from .text import ChineseTextNormalizer


AISHELL_SPLITS = ("train", "dev", "test")
UTTERANCE_PATTERN = re.compile(r"^BAC009(?P<speaker>S\d{4})W\d+$")


def discover_split_audio_paths(
    corpus_root: str | Path, split: str
) -> list[Path]:
    """Return official WAVs while ignoring macOS AppleDouble sidecars."""

    root = Path(corpus_root)
    paths: list[Path] = []
    for path in sorted((root / "data_aishell" / "wav" / split).glob("S*/*.wav")):
        if path.name.startswith("._"):
            continue
        if UTTERANCE_PATTERN.fullmatch(path.stem) is None:
            raise ValueError(f"unexpected WAV filename in AISHELL tree: {path}")
        paths.append(path)
    return paths


def _safe_member_destination(root: Path, name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe tar member path: {name!r}")
    destination = (root / Path(*pure.parts)).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"tar member escapes destination: {name!r}") from exc
    return destination


def extract_tar_safely(archive: str | Path, destination: str | Path) -> int:
    """Extract regular files/directories while rejecting links and traversal."""

    source = Path(archive)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(source, "r:*") as bundle:
        for member in bundle:
            target = _safe_member_destination(root, member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsupported tar member type: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"failed to read tar member: {member.name}")
            temporary = target.with_name(f".{target.name}.extracting")
            with temporary.open("wb") as output:
                while chunk := extracted.read(8 * 1024 * 1024):
                    output.write(chunk)
            temporary.replace(target)
            count += 1
    return count


def extract_nested_wav_archives(corpus_root: str | Path) -> int:
    """Expand the per-speaker archives shipped inside `data_aishell.tgz`."""

    root = Path(corpus_root)
    wav_root = root / "data_aishell" / "wav"
    archives = sorted(
        path
        for path in wav_root.glob("*.tar.gz")
        if not path.name.startswith("._")
    )
    # Accept older/local repacks that place archives one directory deeper.
    archives.extend(
        sorted(
            path
            for path in wav_root.glob("*/*.tar.gz")
            if not path.name.startswith("._")
        )
    )
    for archive in archives:
        extract_tar_safely(archive, wav_root)
    return len(archives)


def read_transcripts(path: str | Path) -> dict[str, str]:
    """Read the official space-separated transcript table."""

    transcripts: dict[str, str] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.strip().split(maxsplit=1)
        if not fields:
            continue
        if len(fields) != 2 or not fields[1].strip():
            raise ValueError(f"invalid transcript at line {line_number}")
        utterance_id, transcript = fields
        if utterance_id in transcripts:
            raise ValueError(f"duplicate transcript id: {utterance_id}")
        transcripts[utterance_id] = transcript.strip()
    if not transcripts:
        raise ValueError("transcript table is empty")
    return transcripts


def read_speaker_info(path: str | Path) -> dict[str, str]:
    """Read official numeric speaker IDs and M/F labels."""

    speakers: dict[str, str] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) != 2 or fields[1] not in {"M", "F"}:
            raise ValueError(f"invalid speaker.info line {line_number}")
        speaker_id = f"S{int(fields[0]):04d}"
        if speaker_id in speakers:
            raise ValueError(f"duplicate speaker metadata: {speaker_id}")
        speakers[speaker_id] = fields[1]
    if not speakers:
        raise ValueError("speaker.info is empty")
    return speakers


def _speaker_from_path(utterance_id: str, audio_path: Path) -> str:
    match = UTTERANCE_PATTERN.fullmatch(utterance_id)
    if match is None:
        raise ValueError(f"unexpected AISHELL utterance id: {utterance_id}")
    speaker = match.group("speaker")
    if audio_path.parent.name != speaker:
        raise ValueError(
            f"speaker mismatch for {utterance_id}: folder={audio_path.parent.name}"
        )
    return speaker


def inspect_audio(path: str | Path) -> dict[str, float | bool]:
    """Stream one file and compute finite/level/clipping audit fields."""

    sample_count = 0
    square_sum = 0.0
    peak = 0.0
    clipped = 0
    finite = True
    for block in sf.blocks(
        path,
        blocksize=65_536,
        dtype="float32",
        always_2d=False,
    ):
        values = np.asarray(block, dtype=np.float64)
        finite = finite and bool(np.all(np.isfinite(values)))
        if not finite:
            continue
        absolute = np.abs(values)
        sample_count += values.size
        square_sum += float(np.sum(values * values))
        peak = max(peak, float(np.max(absolute, initial=0.0)))
        clipped += int(np.count_nonzero(absolute >= 0.999))
    if sample_count == 0:
        return {
            "finite": finite,
            "peak_abs": 0.0,
            "rms_dbfs": float("-inf"),
            "clipped_fraction": 0.0,
        }
    rms = float(np.sqrt(square_sum / sample_count))
    return {
        "finite": finite,
        "peak_abs": peak,
        "rms_dbfs": 20.0 * np.log10(max(rms, np.finfo(float).tiny)),
        "clipped_fraction": clipped / sample_count,
    }


def build_split_manifest(
    corpus_root: str | Path,
    split: str,
    transcripts: Mapping[str, str],
    *,
    minimum_duration_seconds: float = 0.5,
    maximum_duration_seconds: float = 25.0,
    speaker_genders: Mapping[str, str] | None = None,
    maximum_clipped_fraction: float = 0.01,
    normalizer: ChineseTextNormalizer | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one filtered official-split manifest and an exclusion audit."""

    if split not in AISHELL_SPLITS:
        raise ValueError(f"unknown AISHELL split: {split}")
    root = Path(corpus_root).resolve()
    split_root = root / "data_aishell" / "wav" / split
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)
    if not 0 < minimum_duration_seconds < maximum_duration_seconds:
        raise ValueError("invalid duration range")
    if not 0 <= maximum_clipped_fraction <= 1:
        raise ValueError("maximum_clipped_fraction must be in [0, 1]")

    rows: list[dict[str, Any]] = []
    text_normalizer = normalizer or ChineseTextNormalizer()
    excluded: dict[str, list[str]] = {
        "missing_transcript": [],
        "sample_rate": [],
        "channels": [],
        "duration": [],
        "nonfinite": [],
        "silence": [],
        "clipping": [],
    }
    for audio_path in discover_split_audio_paths(root, split):
        utterance_id = audio_path.stem
        transcript_raw = transcripts.get(utterance_id)
        if transcript_raw is None:
            excluded["missing_transcript"].append(utterance_id)
            continue
        speaker_id = _speaker_from_path(utterance_id, audio_path)
        information = sf.info(audio_path)
        if information.samplerate != 16_000:
            excluded["sample_rate"].append(utterance_id)
            continue
        if information.channels != 1:
            excluded["channels"].append(utterance_id)
            continue
        duration = information.frames / information.samplerate
        if not minimum_duration_seconds <= duration <= maximum_duration_seconds:
            excluded["duration"].append(utterance_id)
            continue
        audio_audit = inspect_audio(audio_path)
        if not audio_audit["finite"]:
            excluded["nonfinite"].append(utterance_id)
            continue
        if float(audio_audit["peak_abs"]) <= np.finfo(np.float32).tiny:
            excluded["silence"].append(utterance_id)
            continue
        if float(audio_audit["clipped_fraction"]) > maximum_clipped_fraction:
            excluded["clipping"].append(utterance_id)
            continue
        transcript_normalized = text_normalizer.normalize(transcript_raw)
        if not transcript_normalized:
            raise ValueError(f"empty normalized transcript: {utterance_id}")
        rows.append(
            {
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "gender": None
                if speaker_genders is None
                else speaker_genders.get(speaker_id),
                "split": split,
                "audio_path": audio_path.relative_to(root).as_posix(),
                "transcript_raw": transcript_raw,
                "transcript": transcript_normalized,
                "duration_seconds": duration,
                "sample_rate": information.samplerate,
                "channels": information.channels,
                "frames": information.frames,
                "peak_abs": audio_audit["peak_abs"],
                "rms_dbfs": audio_audit["rms_dbfs"],
                "clipped_fraction": audio_audit["clipped_fraction"],
            }
        )
    if not rows:
        raise ValueError(f"no eligible audio found for split {split}")
    audit = {
        "split": split,
        "utterances": len(rows),
        "speakers": len({row["speaker_id"] for row in rows}),
        "gender_utterances": {
            gender: sum(row["gender"] == gender for row in rows)
            for gender in ("F", "M")
        },
        "hours": sum(float(row["duration_seconds"]) for row in rows) / 3600.0,
        "manifest_sha256": canonical_sha256(rows),
        "excluded": excluded,
        "excluded_counts": {key: len(value) for key, value in excluded.items()},
    }
    return rows, audit


def _as_clean_utterance(row: Mapping[str, Any]) -> CleanUtterance:
    return CleanUtterance(
        utterance_id=str(row["utterance_id"]),
        speaker_id=str(row["speaker_id"]),
        audio_path=str(row["audio_path"]),
        transcript=str(row["transcript"]),
        duration_seconds=float(row["duration_seconds"]),
        domain="AISHELL-1",
    )


def prepare_manifests(
    corpus_root: str | Path,
    output_root: str | Path,
    *,
    train_subset_hours: float = 20.0,
    seed: int = 2026,
    minimum_duration_seconds: float = 0.5,
    maximum_duration_seconds: float = 25.0,
    maximum_clipped_fraction: float = 0.01,
    normalizer: ChineseTextNormalizer | None = None,
    dev_model_utterances: int = 1_000,
    dev_frontend_utterances: int = 500,
    test_reverb_utterances: int = 1_000,
    measured_rir_test_utterances: int = 500,
) -> dict[str, Any]:
    """Create official manifests, a balanced train subset, and a full audit."""

    root = Path(corpus_root).resolve()
    transcript_path = (
        root / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt"
    )
    transcripts = read_transcripts(transcript_path)
    speaker_info_path = root / "resource_aishell" / "speaker.info"
    speaker_genders = (
        read_speaker_info(speaker_info_path) if speaker_info_path.is_file() else None
    )
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_audits: dict[str, dict[str, Any]] = {}
    for split in AISHELL_SPLITS:
        rows, audit = build_split_manifest(
            root,
            split,
            transcripts,
            minimum_duration_seconds=minimum_duration_seconds,
            maximum_duration_seconds=maximum_duration_seconds,
            speaker_genders=speaker_genders,
            maximum_clipped_fraction=maximum_clipped_fraction,
            normalizer=normalizer,
        )
        split_rows[split] = rows
        split_audits[split] = audit

    validate_disjoint_groups(split_rows, group_field="speaker_id")
    discovered_audio_ids: list[str] = []
    for split in AISHELL_SPLITS:
        discovered_audio_ids.extend(
            path.stem
            for path in discover_split_audio_paths(root, split)
        )
    if len(discovered_audio_ids) != len(set(discovered_audio_ids)):
        raise ValueError("duplicate AISHELL utterance ids across official splits")
    audio_without_transcript = sorted(set(discovered_audio_ids) - set(transcripts))
    transcript_without_audio = sorted(set(transcripts) - set(discovered_audio_ids))
    # A WAV without a transcript cannot participate in supervised adaptation or
    # CER scoring, but is a corpus-quality exclusion rather than a reason to
    # discard the otherwise valid official split.  The per-split audit already
    # records those IDs under ``missing_transcript``.  A transcript without its
    # audio counterpart, however, would indicate an incomplete extraction and
    # must stop the run.
    if transcript_without_audio:
        raise ValueError(
            "AISHELL audio/transcript mismatch: "
            f"audio_without_transcript={audio_without_transcript[:5]}, "
            f"transcript_without_audio={transcript_without_audio[:5]}"
        )
    train_subset = select_speaker_balanced_duration_subset(
        (_as_clean_utterance(row) for row in split_rows["train"]),
        target_hours=train_subset_hours,
        seed=seed,
    )
    lookup = {row["utterance_id"]: row for row in split_rows["train"]}
    subset_rows = [lookup[row.utterance_id] for row in train_subset]

    dev_model = select_speaker_balanced_count_subset(
        (_as_clean_utterance(row) for row in split_rows["dev"]),
        count=dev_model_utterances,
        seed=seed,
    )
    dev_model_ids = {row.utterance_id for row in dev_model}
    dev_remaining = [
        _as_clean_utterance(row)
        for row in split_rows["dev"]
        if row["utterance_id"] not in dev_model_ids
    ]
    dev_frontend = select_speaker_balanced_count_subset(
        dev_remaining,
        count=dev_frontend_utterances,
        seed=seed + 1,
    )
    test_reverb = select_speaker_balanced_count_subset(
        (_as_clean_utterance(row) for row in split_rows["test"]),
        count=test_reverb_utterances,
        seed=seed + 2,
    )
    if measured_rir_test_utterances > test_reverb_utterances:
        raise ValueError(
            "measured RIR test count cannot exceed test reverb count"
        )
    measured_rir = select_speaker_balanced_count_subset(
        test_reverb,
        count=measured_rir_test_utterances,
        seed=seed + 3,
    )

    def recover(values: Iterable[CleanUtterance], split: str) -> list[dict[str, Any]]:
        rows_by_id = {row["utterance_id"]: row for row in split_rows[split]}
        return [rows_by_id[value.utterance_id] for value in values]

    derived_rows = {
        "dev_model": recover(dev_model, "dev"),
        "dev_frontend": recover(dev_frontend, "dev"),
        "test_reverb": recover(test_reverb, "test"),
        "test_measured_rir": recover(measured_rir, "test"),
    }

    destination = Path(output_root)
    for split, rows in split_rows.items():
        write_jsonl_atomic(destination / f"aishell1_{split}.jsonl", rows)
    subset_name = f"aishell1_train_{train_subset_hours:g}h.jsonl"
    write_jsonl_atomic(destination / subset_name, subset_rows)
    for name, rows in derived_rows.items():
        write_jsonl_atomic(destination / f"aishell1_{name}.jsonl", rows)

    audit = {
        "schema_version": 1,
        "corpus_root": str(root),
        "seed": seed,
        "duration_filter_seconds": [
            minimum_duration_seconds,
            maximum_duration_seconds,
        ],
        "maximum_clipped_fraction": maximum_clipped_fraction,
        "splits": split_audits,
        "speaker_split_disjoint": True,
        "speaker_metadata_loaded": speaker_genders is not None,
        "train_subset": {
            "path": subset_name,
            "target_hours": train_subset_hours,
            "utterances": len(subset_rows),
            "speakers": len({row["speaker_id"] for row in subset_rows}),
            "hours": sum(float(row["duration_seconds"]) for row in subset_rows)
            / 3600.0,
            "manifest_sha256": canonical_sha256(subset_rows),
        },
        "derived_subsets": {
            name: {
                "path": f"aishell1_{name}.jsonl",
                "utterances": len(rows),
                "speakers": len({row["speaker_id"] for row in rows}),
                "hours": sum(float(row["duration_seconds"]) for row in rows)
                / 3600.0,
                "manifest_sha256": canonical_sha256(rows),
            }
            for name, rows in derived_rows.items()
        },
        "dev_model_frontend_disjoint": not (
            {row["utterance_id"] for row in derived_rows["dev_model"]}
            & {row["utterance_id"] for row in derived_rows["dev_frontend"]}
        ),
        "audio_transcript_one_to_one": not (
            audio_without_transcript or transcript_without_audio
        ),
        "audio_without_transcript": audio_without_transcript,
        "transcript_without_audio": transcript_without_audio,
        "discovered_audio_files": len(discovered_audio_ids),
    }
    return audit
