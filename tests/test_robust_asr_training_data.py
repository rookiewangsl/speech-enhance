from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from robust_asr.training.data import (
    WhisperAdaptationDataset,
    WhisperBatchCollator,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    corpus = tmp_path / "corpus"
    audio = corpus / "wav" / "sample.wav"
    audio.parent.mkdir(parents=True)
    time = np.arange(1_600, dtype=np.float32) / 16_000
    sf.write(audio, 0.05 * np.sin(2 * np.pi * 440 * time), 16_000)
    manifest = tmp_path / "train.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "utterance_id": "utt-1",
                "speaker_id": "speaker-1",
                "audio_path": "wav/sample.wav",
                "transcript": "测试语音",
            }
        ],
    )

    rir_root = tmp_path / "rir"
    rir_path = rir_root / "train" / "train-rir.npz"
    rir_path.parent.mkdir(parents=True)
    full = np.zeros((4, 81), dtype=np.float64)
    full[:, 0] = (1.0, 0.9, 0.8, 0.7)
    full[:, 80] = 0.2
    np.savez_compressed(rir_path, full=full, direct=full[:, :1])
    rir_manifest = rir_root / "train.jsonl"
    _write_jsonl(
        rir_manifest,
        [
            {
                "rir_id": "train-rir",
                "split": "train",
                "room_id": "train-room",
                "path": "train/train-rir.npz",
                "file_sha256": _sha256(rir_path),
                "full_shape": [4, 81],
                "target_rt60_seconds": 0.6,
            }
        ],
    )
    return manifest, corpus, rir_manifest, rir_root


def test_clean_dataset_uses_same_level_protocol_as_reverb(tmp_path: Path) -> None:
    manifest, corpus, _, _ = _fixture_tree(tmp_path)
    dataset = WhisperAdaptationDataset(
        manifest_path=manifest,
        corpus_root=corpus,
        mode="clean",
    )

    row = dataset[0]

    assert row["condition"] == "clean"
    assert row["rir_id"] is None
    assert row["waveform"].shape == (1_600,)
    assert row["reference_rms_dbfs"] == pytest.approx(-25.0, abs=0.02)
    assert dataset.audit()["manifest_sha256"] == dataset.manifest_sha256


def test_mct_dataset_convolves_train_rir_deterministically(tmp_path: Path) -> None:
    manifest, corpus, rir_manifest, rir_root = _fixture_tree(tmp_path)
    first = WhisperAdaptationDataset(
        manifest_path=manifest,
        corpus_root=corpus,
        mode="mct",
        rir_manifest_path=rir_manifest,
        rir_root=rir_root,
        reverb_probability=1.0,
        seed=7,
    )
    second = WhisperAdaptationDataset(
        manifest_path=manifest,
        corpus_root=corpus,
        mode="mct",
        rir_manifest_path=rir_manifest,
        rir_root=rir_root,
        reverb_probability=1.0,
        seed=7,
    )

    assert first.decision(0) == second.decision(0) == ("raw_reverb", "train-rir")
    row = first[0]
    assert row["waveform"].shape == (1_680,)
    assert row["target_rt60_seconds"] == 0.6
    assert row["room_id"] == "train-room"


def test_mct_rejects_nontrain_rir_split(tmp_path: Path) -> None:
    manifest, corpus, rir_manifest, rir_root = _fixture_tree(tmp_path)
    rows = [json.loads(rir_manifest.read_text(encoding="utf-8"))]
    rows[0]["split"] = "dev"
    _write_jsonl(rir_manifest, rows)

    with pytest.raises(ValueError, match="train RIR split"):
        WhisperAdaptationDataset(
            manifest_path=manifest,
            corpus_root=corpus,
            mode="mct",
            rir_manifest_path=rir_manifest,
            rir_root=rir_root,
        )


def test_epoch_changes_are_reproducible() -> None:
    # This test uses many IDs so that the probability of an unchanged vector is
    # negligible while still asserting exact cross-instance reproducibility.
    from robust_asr.manifest import choose_mct_condition

    ids = [f"utt-{index}" for index in range(100)]
    epoch_zero = [
        choose_mct_condition(value, epoch=0, seed=2026) for value in ids
    ]
    repeated = [
        choose_mct_condition(value, epoch=0, seed=2026) for value in ids
    ]
    epoch_one = [
        choose_mct_condition(value, epoch=1, seed=2026) for value in ids
    ]

    assert epoch_zero == repeated
    assert epoch_zero != epoch_one


def test_whisper_collator_masks_padding_and_removes_decoder_start() -> None:
    torch = pytest.importorskip("torch")

    class FeatureExtractor:
        def __call__(self, waveforms, **kwargs):
            assert kwargs["sampling_rate"] == 16_000
            assert kwargs["padding"] == "max_length"
            return {"input_features": torch.ones((len(waveforms), 80, 3000))}

    class Tokenizer:
        bos_token_id = 1

        def __call__(self, texts, **kwargs):
            assert kwargs["padding"] is True
            input_ids = torch.tensor([[1, 5, 6], [1, 7, 0]])
            attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class Processor:
        feature_extractor = FeatureExtractor()
        tokenizer = Tokenizer()

    rows = [
        {
            "utterance_id": f"utt-{index}",
            "speaker_id": "s",
            "waveform": np.ones(160 + index, dtype=np.float32),
            "sample_rate": 16_000,
            "transcript": text,
            "condition": "clean",
            "rir_id": None,
            "room_id": None,
            "target_rt60_seconds": None,
        }
        for index, text in enumerate(("测试", "语音"))
    ]

    batch = WhisperBatchCollator(Processor())(rows)

    assert batch["input_features"].shape == (2, 80, 3000)
    assert batch["labels"].tolist() == [[5, 6], [7, -100]]
    assert [row["utterance_id"] for row in batch["metadata"]] == ["utt-0", "utt-1"]
