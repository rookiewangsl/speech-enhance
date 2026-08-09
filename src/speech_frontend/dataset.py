"""Dataset discovery, safe extraction, splitting, and resampling."""

from __future__ import annotations

import json
import random
import shutil
import stat
import zipfile
from dataclasses import dataclass
from math import gcd
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from speech_frontend.audio import (
    AudioData,
    read_audio,
    validate_aligned_pair,
    write_audio,
)


@dataclass(frozen=True)
class PairedUtterance:
    """Paths and grouping metadata for one aligned utterance pair."""

    utterance_id: str
    speaker_id: str
    clean_path: Path
    noisy_path: Path


def _wav_map(directory: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        if path.stem in paths:
            raise ValueError(f"duplicate WAV stem: {path.stem}")
        paths[path.stem] = path
    return paths


def discover_voicebank_pairs(
    clean_directory: str | Path,
    noisy_directory: str | Path,
) -> list[PairedUtterance]:
    """Match VoiceBank clean and noisy WAV files by filename stem."""

    clean_paths = _wav_map(Path(clean_directory))
    noisy_paths = _wav_map(Path(noisy_directory))
    if not clean_paths or not noisy_paths:
        raise ValueError("clean and noisy directories must contain WAV files")

    missing_noisy = sorted(clean_paths.keys() - noisy_paths.keys())
    missing_clean = sorted(noisy_paths.keys() - clean_paths.keys())
    if missing_noisy or missing_clean:
        raise ValueError(
            "clean/noisy filenames do not match; "
            f"missing noisy={missing_noisy[:3]}, "
            f"missing clean={missing_clean[:3]}"
        )

    records: list[PairedUtterance] = []
    for utterance_id in sorted(clean_paths):
        speaker_id = utterance_id.split("_", maxsplit=1)[0]
        if speaker_id == utterance_id:
            raise ValueError(
                f"cannot derive speaker id from utterance: {utterance_id}"
            )
        records.append(
            PairedUtterance(
                utterance_id=utterance_id,
                speaker_id=speaker_id,
                clean_path=clean_paths[utterance_id],
                noisy_path=noisy_paths[utterance_id],
            )
        )
    return records


def split_pairs_by_speaker(
    records: list[PairedUtterance],
    *,
    dev_fraction: float = 0.4,
    seed: int = 20260724,
) -> tuple[list[PairedUtterance], list[PairedUtterance]]:
    """Create deterministic dev/holdout splits with disjoint speakers."""

    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between zero and one")
    speakers = sorted({record.speaker_id for record in records})
    if len(speakers) < 2:
        raise ValueError("speaker-group split requires at least two speakers")

    random.Random(seed).shuffle(speakers)
    dev_count = round(len(speakers) * dev_fraction)
    dev_count = min(max(dev_count, 1), len(speakers) - 1)
    dev_speakers = set(speakers[:dev_count])
    dev = [record for record in records if record.speaker_id in dev_speakers]
    holdout = [
        record for record in records if record.speaker_id not in dev_speakers
    ]
    return dev, holdout


def split_pairs_by_speaker_count(
    records: list[PairedUtterance],
    *,
    first_speaker_count: int,
    seed: int = 20260724,
) -> tuple[list[PairedUtterance], list[PairedUtterance]]:
    """Split pairs into two deterministic speaker-disjoint partitions.

    Unlike :func:`split_pairs_by_speaker`, this API fixes the exact number of
    speakers in the first partition.  It is used by the full VoiceBank protocol
    to keep a reproducible 20-speaker development / 8-speaker validation split.
    """

    speakers = sorted({record.speaker_id for record in records})
    if not 0 < first_speaker_count < len(speakers):
        raise ValueError(
            "first_speaker_count must be between zero and the total speaker "
            "count"
        )
    random.Random(seed).shuffle(speakers)
    first_speakers = set(speakers[:first_speaker_count])
    first = [record for record in records if record.speaker_id in first_speakers]
    second = [record for record in records if record.speaker_id not in first_speakers]
    return first, second


def sample_manifest_rows_by_speaker(
    rows: list[dict[str, Any]],
    *,
    items_per_speaker: int,
    seed: int = 20260724,
) -> list[dict[str, Any]]:
    """Take a reproducible, balanced utterance sample from every speaker."""

    if items_per_speaker <= 0:
        raise ValueError("items_per_speaker must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        speaker_id = row.get("speaker_id")
        if not isinstance(speaker_id, str) or not speaker_id:
            raise ValueError("each manifest row must have a speaker_id")
        grouped.setdefault(speaker_id, []).append(row)
    if not grouped:
        raise ValueError("manifest has no rows")

    sampled: list[dict[str, Any]] = []
    for speaker_id in sorted(grouped):
        candidates = sorted(grouped[speaker_id], key=lambda row: str(row["id"]))
        random.Random(f"{seed}:{speaker_id}").shuffle(candidates)
        sampled.extend(candidates[:items_per_speaker])
    return sampled


def resample_aligned_pair(
    clean: AudioData,
    noisy: AudioData,
    *,
    target_sample_rate: int = 16_000,
) -> tuple[AudioData, AudioData]:
    """Resample an aligned pair with identical polyphase parameters."""

    validate_aligned_pair(clean, noisy)
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")
    if clean.sample_rate == target_sample_rate:
        return clean, noisy

    common_divisor = gcd(clean.sample_rate, target_sample_rate)
    up = target_sample_rate // common_divisor
    down = clean.sample_rate // common_divisor
    clean_samples = resample_poly(clean.samples, up, down)
    noisy_samples = resample_poly(noisy.samples, up, down)
    result = (
        AudioData(clean_samples.astype(np.float32), target_sample_rate),
        AudioData(noisy_samples.astype(np.float32), target_sample_rate),
    )
    validate_aligned_pair(*result)
    return result


def prepare_pair(
    record: PairedUtterance,
    clean_output: str | Path,
    noisy_output: str | Path,
    *,
    target_sample_rate: int = 16_000,
) -> dict[str, Any]:
    """Validate, resample, and write one paired utterance."""

    clean, noisy = resample_aligned_pair(
        read_audio(record.clean_path),
        read_audio(record.noisy_path),
        target_sample_rate=target_sample_rate,
    )
    clean_path = Path(clean_output)
    noisy_path = Path(noisy_output)
    write_audio(clean_path, clean)
    write_audio(noisy_path, noisy)
    return {
        "id": record.utterance_id,
        "speaker_id": record.speaker_id,
        "clean": clean_path.as_posix(),
        "noisy": noisy_path.as_posix(),
        "sample_rate": target_sample_rate,
        "num_samples": int(clean.samples.size),
    }


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Atomically write one JSON object per line."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)


def safe_extract_wav_zip(
    archive: str | Path,
    destination: str | Path,
    *,
    maximum_files: int = 20_000,
    maximum_uncompressed_bytes: int = 20 * 1024**3,
) -> Path:
    """Extract only WAV files while rejecting path and archive attacks."""

    archive_path = Path(archive)
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(
            f"extraction destination already exists: {destination_path}"
        )

    with zipfile.ZipFile(archive_path) as source:
        members = [member for member in source.infolist() if not member.is_dir()]
        if len(members) > maximum_files:
            raise ValueError("archive contains too many files")
        total_size = sum(member.file_size for member in members)
        if total_size > maximum_uncompressed_bytes:
            raise ValueError("archive is too large after extraction")

        validated: list[tuple[zipfile.ZipInfo, Path]] = []
        for member in members:
            relative = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or stat.S_ISLNK(mode)
            ):
                raise ValueError(f"unsafe ZIP member: {member.filename}")
            if relative.suffix.lower() != ".wav":
                raise ValueError(f"unexpected non-WAV file: {member.filename}")
            validated.append((member, Path(*relative.parts)))

        temporary = destination_path.with_name(destination_path.name + ".part")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            for member, relative in validated:
                output = temporary / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as input_stream:
                    with output.open("wb") as output_stream:
                        shutil.copyfileobj(input_stream, output_stream)
            temporary.replace(destination_path)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return destination_path
