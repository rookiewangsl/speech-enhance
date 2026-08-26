from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from robust_asr.baseline import run_frozen_baseline, select_speaker_balanced_count
from robust_asr.dereverb.frontend import apply_frontend
from robust_asr.download import sha256_file
from robust_asr.manifest import write_jsonl_atomic
from robust_asr.text import ChineseTextNormalizer


class EchoTranscriber:
    model_id = "test/echo"
    device = "cpu"

    def transcribe(self, audio: np.ndarray, *, sample_rate: int = 16_000) -> str:
        assert sample_rate == 16_000
        assert audio.ndim == 1
        return "测 试"


def test_speaker_balanced_count_is_deterministic() -> None:
    rows = [
        {"utterance_id": "u1", "speaker_id": "s1"},
        {"utterance_id": "u2", "speaker_id": "s1"},
        {"utterance_id": "u3", "speaker_id": "s2"},
    ]
    first = select_speaker_balanced_count(rows, limit=2, seed=7)
    second = select_speaker_balanced_count(reversed(rows), limit=2, seed=7)
    assert first == second
    assert len({row["speaker_id"] for row in first}) == 2


def test_nara_frontends_preserve_mono_length() -> None:
    pytest.importorskip("nara_wpe")
    rng = np.random.default_rng(3)
    audio = rng.normal(scale=0.01, size=(4, 8_000))
    raw = apply_frontend(audio, "raw")
    single = apply_frontend(audio, "s_wpe_10", backend="nara_wpe")
    multi = apply_frontend(audio, "m_wpe_10", backend="nara_wpe")
    assert raw.shape == single.shape == multi.shape == (8_000,)
    assert np.isfinite(single).all()
    assert np.isfinite(multi).all()


def test_end_to_end_baseline_with_fake_transcriber(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    wav = corpus / "data_aishell/wav/test/S0001/BAC009S0001W0001.wav"
    wav.parent.mkdir(parents=True)
    time = np.arange(8_000, dtype=np.float32) / 16_000
    sf.write(wav, 0.05 * np.sin(2 * np.pi * 220 * time), 16_000)
    manifest = tmp_path / "test.jsonl"
    write_jsonl_atomic(
        manifest,
        [
            {
                "utterance_id": "BAC009S0001W0001",
                "speaker_id": "S0001",
                "audio_path": wav.relative_to(corpus).as_posix(),
                "transcript": "测试",
            }
        ],
    )
    rir_root = tmp_path / "rir"
    (rir_root / "smoke").mkdir(parents=True)
    full = np.zeros((4, 64), dtype=np.float32)
    full[:, 0] = 1.0
    np.savez_compressed(rir_root / "smoke/r0.npz", full=full, direct=full)
    rir_manifest = rir_root / "smoke.jsonl"
    rir_path = rir_root / "smoke/r0.npz"
    write_jsonl_atomic(
        rir_manifest,
        [
            {
                "rir_id": "r0",
                "room_id": "room0",
                "scene": {"source_array_distance_m": 2.0},
                "path": "smoke/r0.npz",
                "file_sha256": sha256_file(rir_path),
                "full_shape": [4, 64],
                "target_rt60_seconds": 0.2,
                "measured_rt60_seconds": [0.2] * 4,
                "drr_db": [10.0] * 4,
            }
        ],
    )
    output = tmp_path / "results.jsonl"
    summary = run_frozen_baseline(
        manifest_path=manifest,
        corpus_root=corpus,
        rir_manifest_path=rir_manifest,
        rir_root=rir_root,
        output_path=output,
        transcriber=EchoTranscriber(),
        limit=1,
        frontends=("raw",),
        rt60_seconds=(0.2,),
        normalizer=ChineseTextNormalizer(traditional_to_simplified=False),
    )
    assert summary["result_rows"] == 2
    assert all(row["cer"] == 0 for row in summary["conditions"])
    stored = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["frontend"] for row in stored} == {"clean", "raw"}
