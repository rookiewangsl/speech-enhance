from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from robust_asr.aishell import (
    build_split_manifest,
    extract_nested_wav_archives,
    extract_tar_safely,
    prepare_manifests,
    read_speaker_info,
    read_transcripts,
)
from robust_asr.download import finalize_download
from robust_asr.paths import data_root, initialize_data_root
from robust_asr.text import ChineseTextNormalizer


def _write_utterance(
    corpus_root: Path,
    split: str,
    speaker: str,
    utterance_id: str,
    *,
    seconds: float = 0.75,
) -> None:
    path = (
        corpus_root
        / "data_aishell"
        / "wav"
        / split
        / speaker
        / f"{utterance_id}.wav"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(round(16_000 * seconds), dtype=np.float32) / 16_000
    samples = 0.05 * np.sin(2 * np.pi * 220 * time)
    sf.write(path, samples, 16_000, subtype="PCM_16")


def test_read_transcripts_and_build_manifest(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    transcript = corpus / "transcripts.txt"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("BAC009S0001W0001 語 音，識 別\n", encoding="utf-8")
    utterance_id = "BAC009S0001W0001"
    _write_utterance(corpus, "train", "S0001", utterance_id)
    sidecar = (
        corpus
        / "data_aishell"
        / "wav"
        / "train"
        / "S0001"
        / f"._{utterance_id}.wav"
    )
    sidecar.write_bytes(b"AppleDouble")

    transcripts = read_transcripts(transcript)
    rows, audit = build_split_manifest(
        corpus,
        "train",
        transcripts,
        normalizer=ChineseTextNormalizer(
            converter=lambda value: value.translate(
                str.maketrans({"語": "语", "識": "识", "別": "别"})
            )
        ),
    )
    assert rows[0]["transcript"] == "语音识别"
    assert rows[0]["audio_path"].endswith(f"S0001/{utterance_id}.wav")
    assert audit["utterances"] == 1
    assert audit["hours"] == pytest.approx(0.75 / 3600.0)


def test_read_official_speaker_info(tmp_path: Path) -> None:
    path = tmp_path / "speaker.info"
    path.write_text("0002 M\n0764 F\n", encoding="utf-8")

    assert read_speaker_info(path) == {"S0002": "M", "S0764": "F"}


def test_safe_tar_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("../escape.txt")
        payload = b"escape"
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe tar member"):
        extract_tar_safely(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_nested_official_speaker_archive_expands_split_path(tmp_path: Path) -> None:
    wav_root = tmp_path / "data_aishell" / "wav"
    wav_root.mkdir(parents=True)
    archive = wav_root / "S0001.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        payload = b"wav-bytes"
        member = tarfile.TarInfo("train/S0001/BAC009S0001W0001.wav")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))

    assert extract_nested_wav_archives(tmp_path) == 1
    assert (
        wav_root / "train" / "S0001" / "BAC009S0001W0001.wav"
    ).read_bytes() == b"wav-bytes"


def test_prepare_manifests_rejects_speaker_leakage(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    transcript = (
        corpus
        / "data_aishell"
        / "transcript"
        / "aishell_transcript_v0.8.txt"
    )
    transcript.parent.mkdir(parents=True)
    identifiers = [
        "BAC009S0001W0001",
        "BAC009S0001W0002",
        "BAC009S0003W0001",
    ]
    transcript.write_text(
        "\n".join(f"{value} 测 试" for value in identifiers) + "\n",
        encoding="utf-8",
    )
    _write_utterance(corpus, "train", "S0001", identifiers[0])
    _write_utterance(corpus, "dev", "S0001", identifiers[1])
    _write_utterance(corpus, "test", "S0003", identifiers[2])
    with pytest.raises(ValueError, match="speaker_id leakage"):
        prepare_manifests(
            corpus,
            tmp_path / "manifests",
            train_subset_hours=0.0001,
            normalizer=ChineseTextNormalizer(traditional_to_simplified=False),
        )


def test_prepare_manifests_creates_disjoint_derived_subsets(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    transcript_path = (
        corpus
        / "data_aishell"
        / "transcript"
        / "aishell_transcript_v0.8.txt"
    )
    transcript_path.parent.mkdir(parents=True)
    identifiers: list[str] = []
    for split, speaker, count in (
        ("train", "S0001", 2),
        ("dev", "S0002", 6),
        ("test", "S0003", 4),
    ):
        for index in range(count):
            utterance_id = f"BAC009{speaker}W{index:04d}"
            identifiers.append(utterance_id)
            _write_utterance(corpus, split, speaker, utterance_id)
    transcript_path.write_text(
        "\n".join(f"{value} 测 试" for value in identifiers) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "manifests"
    audit = prepare_manifests(
        corpus,
        output,
        train_subset_hours=0.0001,
        dev_model_utterances=2,
        dev_frontend_utterances=2,
        test_reverb_utterances=3,
        measured_rir_test_utterances=2,
        normalizer=ChineseTextNormalizer(traditional_to_simplified=False),
    )

    assert audit["dev_model_frontend_disjoint"] is True
    assert audit["derived_subsets"]["dev_model"]["utterances"] == 2
    model_ids = {
        json.loads(line)["utterance_id"]
        for line in (output / "aishell1_dev_model.jsonl").read_text().splitlines()
    }
    frontend_ids = {
        json.loads(line)["utterance_id"]
        for line in (output / "aishell1_dev_frontend.jsonl").read_text().splitlines()
    }
    assert model_ids.isdisjoint(frontend_ids)


def test_prepare_manifests_audits_audio_without_transcript(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    transcript_path = (
        corpus
        / "data_aishell"
        / "transcript"
        / "aishell_transcript_v0.8.txt"
    )
    transcript_path.parent.mkdir(parents=True)
    identifiers: list[str] = []
    for split, speaker, count in (
        ("train", "S0001", 2),
        ("dev", "S0002", 6),
        ("test", "S0003", 4),
    ):
        for index in range(count):
            utterance_id = f"BAC009{speaker}W{index:04d}"
            identifiers.append(utterance_id)
            _write_utterance(corpus, split, speaker, utterance_id)
    missing_transcript_id = "BAC009S0001W9999"
    _write_utterance(corpus, "train", "S0001", missing_transcript_id)
    transcript_path.write_text(
        "\n".join(f"{value} 测 试" for value in identifiers) + "\n",
        encoding="utf-8",
    )

    audit = prepare_manifests(
        corpus,
        tmp_path / "manifests",
        train_subset_hours=0.0001,
        dev_model_utterances=2,
        dev_frontend_utterances=2,
        test_reverb_utterances=3,
        measured_rir_test_utterances=2,
        normalizer=ChineseTextNormalizer(traditional_to_simplified=False),
    )

    assert audit["audio_transcript_one_to_one"] is False
    assert audit["audio_without_transcript"] == [missing_transcript_id]
    assert audit["splits"]["train"]["excluded"]["missing_transcript"] == [
        missing_transcript_id
    ]


def test_external_layout_and_download_receipt(tmp_path: Path) -> None:
    root = initialize_data_root(tmp_path / "robust")
    assert data_root(root) == root.resolve()
    assert (root / "cache" / "huggingface").is_dir()

    partial = root / "downloads" / "sample.bin.part"
    partial.write_bytes(b"abc")
    receipt = finalize_download(
        partial,
        root / "downloads" / "sample.bin",
        url="https://example.test/sample.bin",
        expected_bytes=3,
    )
    assert receipt.sha256 == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    stored = json.loads(
        (root / "downloads" / "sample.bin.receipt.json").read_text()
    )
    assert stored["bytes"] == 3
