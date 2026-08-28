from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from robust_asr.audit import run_whisper_input_audit
from robust_asr.download import sha256_file
from robust_asr.manifest import write_jsonl_atomic


class EchoTranscriber:
    model_id = "test/echo"
    model_revision = "revision"
    device = "cpu"
    num_beams = 1

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio: np.ndarray, *, sample_rate: int = 16_000) -> str:
        assert sample_rate == 16_000
        assert audio.ndim == 1
        assert np.isfinite(audio).all()
        self.calls += 1
        return "测试"


def test_input_audit_is_matched_and_resumable(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    audio_path = corpus / "wav/u.wav"
    audio_path.parent.mkdir(parents=True)
    sf.write(audio_path, np.ones(1_600, dtype=np.float32) * 0.02, 16_000)
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl_atomic(
        manifest,
        [
            {
                "utterance_id": "u",
                "speaker_id": "s",
                "audio_path": "wav/u.wav",
                "transcript": "测试",
            }
        ],
    )
    rir_root = tmp_path / "rir"
    rir_path = rir_root / "dev/r.npz"
    rir_path.parent.mkdir(parents=True)
    direct = np.zeros((4, 8), dtype=np.float32)
    direct[:, 0] = 1.0
    full = direct.copy()
    full[:, 7] = 0.1
    np.savez_compressed(rir_path, full=full, direct=direct)
    rir_manifest = rir_root / "dev.jsonl"
    write_jsonl_atomic(
        rir_manifest,
        [
            {
                "rir_id": "r",
                "room_id": "room",
                "path": "dev/r.npz",
                "file_sha256": sha256_file(rir_path),
                "full_shape": [4, 8],
                "direct_shape": [4, 8],
                "target_rt60_seconds": 0.2,
            }
        ],
    )
    monkeypatch.setattr(
        "robust_asr.audit.apply_frontend",
        lambda signals, frontend, backend: np.asarray(signals[0]),
    )
    output = tmp_path / "audit.jsonl"
    transcriber = EchoTranscriber()
    conditions = (
        "clean_original",
        "clean_level",
        "direct_raw",
        "direct_m_wpe_10",
    )

    first = run_whisper_input_audit(
        manifest_path=manifest,
        corpus_root=corpus,
        rir_manifest_path=rir_manifest,
        rir_root=rir_root,
        output_path=output,
        transcriber=transcriber,
        limit=1,
        conditions=conditions,
        bootstrap_draws=10,
        checkpoint_every_results=2,
    )
    second = run_whisper_input_audit(
        manifest_path=manifest,
        corpus_root=corpus,
        rir_manifest_path=rir_manifest,
        rir_root=rir_root,
        output_path=output,
        transcriber=transcriber,
        limit=1,
        conditions=conditions,
        bootstrap_draws=10,
    )

    assert first["result_rows"] == 4
    assert first["generated_rows"] == 4
    assert second["resumed_rows"] == 4
    assert second["generated_rows"] == 0
    assert transcriber.calls == 4
    assert {row["condition"] for row in map(json.loads, output.read_text().splitlines())} == set(conditions)
    assert {
        (row["baseline"], row["candidate"])
        for row in first["paired_deltas"]
    } == {
        ("clean_original", "clean_level"),
        ("clean_level", "direct_raw"),
        ("direct_raw", "direct_m_wpe_10"),
    }
