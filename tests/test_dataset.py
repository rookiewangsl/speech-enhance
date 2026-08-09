from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from speech_frontend.audio import AudioData
from speech_frontend.dataset import (
    PairedUtterance,
    discover_voicebank_pairs,
    resample_aligned_pair,
    safe_extract_wav_zip,
    sample_manifest_rows_by_speaker,
    split_pairs_by_speaker,
    split_pairs_by_speaker_count,
)


def touch_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_discover_and_split_pairs_keep_speakers_disjoint(
    tmp_path: Path,
) -> None:
    clean = tmp_path / "clean"
    noisy = tmp_path / "noisy"
    for stem in ("p001_001", "p001_002", "p002_001", "p003_001"):
        touch_wav(clean / f"{stem}.wav")
        touch_wav(noisy / f"{stem}.wav")

    records = discover_voicebank_pairs(clean, noisy)
    dev, holdout = split_pairs_by_speaker(records, seed=4)

    assert len(records) == 4
    assert {item.speaker_id for item in dev}.isdisjoint(
        {item.speaker_id for item in holdout}
    )
    assert {item.utterance_id for item in dev + holdout} == {
        item.utterance_id for item in records
    }


def test_split_pairs_by_speaker_count_is_exact_and_reproducible(
    tmp_path: Path,
) -> None:
    records = []
    for speaker in range(6):
        for utterance in range(2):
            path = tmp_path / f"p{speaker}_{utterance}.wav"
            records.append(
                PairedUtterance(path.stem, f"p{speaker}", path, path)
            )

    first, second = split_pairs_by_speaker_count(
        records, first_speaker_count=4, seed=7
    )
    repeated, _ = split_pairs_by_speaker_count(
        records, first_speaker_count=4, seed=7
    )

    first_speakers = {item.speaker_id for item in first}
    second_speakers = {item.speaker_id for item in second}
    assert len(first_speakers) == 4
    assert len(second_speakers) == 2
    assert first_speakers.isdisjoint(second_speakers)
    assert [item.utterance_id for item in first] == [
        item.utterance_id for item in repeated
    ]


def test_sample_manifest_rows_is_balanced_and_reproducible() -> None:
    rows = [
        {"id": f"{speaker}_{index}", "speaker_id": speaker}
        for speaker, count in (("p1", 4), ("p2", 3), ("p3", 2))
        for index in range(count)
    ]
    sampled = sample_manifest_rows_by_speaker(
        rows, items_per_speaker=2, seed=11
    )
    repeated = sample_manifest_rows_by_speaker(
        rows, items_per_speaker=2, seed=11
    )

    counts = {
        speaker: sum(row["speaker_id"] == speaker for row in sampled)
        for speaker in ("p1", "p2", "p3")
    }
    assert counts == {"p1": 2, "p2": 2, "p3": 2}
    assert sampled == repeated


def test_discover_rejects_unmatched_files(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    noisy = tmp_path / "noisy"
    touch_wav(clean / "p001_001.wav")
    touch_wav(noisy / "p001_002.wav")

    with pytest.raises(ValueError, match="do not match"):
        discover_voicebank_pairs(clean, noisy)


def test_resample_pair_preserves_alignment_and_common_scale() -> None:
    samples = np.linspace(-0.5, 0.5, 4_800, dtype=np.float32)
    clean = AudioData(samples, 48_000)
    noisy = AudioData(samples * 0.5, 48_000)

    clean_16k, noisy_16k = resample_aligned_pair(clean, noisy)

    assert clean_16k.sample_rate == 16_000
    assert clean_16k.samples.shape == noisy_16k.samples.shape == (1_600,)
    np.testing.assert_allclose(
        noisy_16k.samples,
        clean_16k.samples * 0.5,
        atol=1e-6,
    )


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.wav", b"not audio")

    with pytest.raises(ValueError, match="unsafe"):
        safe_extract_wav_zip(archive, tmp_path / "output")


def test_safe_extract_accepts_nested_wav(tmp_path: Path) -> None:
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("voicebank/p001_001.wav", b"audio bytes")

    destination = safe_extract_wav_zip(archive, tmp_path / "output")

    assert (destination / "voicebank" / "p001_001.wav").read_bytes() == (
        b"audio bytes"
    )
