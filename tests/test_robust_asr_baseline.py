from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from robust_asr.baseline import (
    BaselineProgress,
    _paired_deltas,
    _select_rir,
    run_frozen_baseline,
    select_speaker_balanced_count,
)
from robust_asr.dereverb.frontend import apply_frontend
from robust_asr.download import sha256_file
from robust_asr.manifest import write_jsonl_atomic
from robust_asr.text import ChineseTextNormalizer


class EchoTranscriber:
    model_id = "test/echo"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio: np.ndarray, *, sample_rate: int = 16_000) -> str:
        assert sample_rate == 16_000
        assert audio.ndim == 1
        self.calls += 1
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


def test_formal_test_rir_selection_keeps_family_fixed_across_rt60() -> None:
    rows = [
        {
            "rir_id": f"{family}_{rt60}",
            "rir_family_id": family,
            "split": "test",
            "target_rt60_seconds": rt60,
        }
        for family in ("family_a", "family_b", "family_c")
        for rt60 in (0.2, 0.6, 1.0)
    ]

    selected = [
        _select_rir(
            list(reversed(rows)),
            utterance_id="utterance",
            rt60_seconds=rt60,
            seed=2026,
        )
        for rt60 in (0.2, 0.6, 1.0)
    ]

    assert len({row["rir_family_id"] for row in selected}) == 1
    assert [row["target_rt60_seconds"] for row in selected] == [0.2, 0.6, 1.0]


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


def test_end_to_end_baseline_resumes_when_frontends_are_extended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    transcriber = EchoTranscriber()
    monkeypatch.setattr(
        "robust_asr.baseline.apply_frontend",
        lambda audio, frontend, backend: np.asarray(audio[0]),
    )
    events: list[BaselineProgress] = []
    summary = run_frozen_baseline(
        manifest_path=manifest,
        corpus_root=corpus,
        rir_manifest_path=rir_manifest,
        rir_root=rir_root,
        output_path=output,
        transcriber=transcriber,
        limit=1,
        frontends=("raw",),
        rt60_seconds=(0.2,),
        normalizer=ChineseTextNormalizer(traditional_to_simplified=False),
        checkpoint_every_results=2,
        progress_callback=events.append,
    )
    assert summary["result_rows"] == 2
    assert summary["generated_rows"] == 2
    assert summary["raw_robustness"]["utterances"] == 1
    assert summary["raw_robustness"]["by_target_rt60"]["0.2"]["unique_rirs"] == 1
    assert (
        summary["raw_robustness"]["by_target_rt60"]["0.2"]
        ["drr_cer_degradation_spearman"]["spearman_rho"]
        is None
    )
    assert all(row["cer"] == 0 for row in summary["conditions"])
    assert transcriber.calls == 2
    assert [event.stage for event in events] == [
        "start",
        "progress",
        "progress",
        "complete",
    ]

    extended = run_frozen_baseline(
        manifest_path=manifest,
        corpus_root=corpus,
        rir_manifest_path=rir_manifest,
        rir_root=rir_root,
        output_path=output,
        transcriber=transcriber,
        limit=1,
        frontends=("raw", "s_wpe_10"),
        rt60_seconds=(0.2,),
        normalizer=ChineseTextNormalizer(traditional_to_simplified=False),
    )
    assert extended["result_rows"] == 3
    assert extended["resumed_rows"] == 2
    assert extended["generated_rows"] == 1
    assert transcriber.calls == 3
    stored = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["frontend"] for row in stored} == {
        "clean",
        "raw",
        "s_wpe_10",
    }


def test_paired_deltas_compare_multichannel_with_both_single_channel_controls() -> None:
    results = [
        {
            "utterance_id": "u1",
            "frontend": frontend,
            "target_rt60_seconds": 0.6,
            "reference": "测试语音",
            "hypothesis": hypothesis,
        }
        for frontend, hypothesis in (
            ("raw", "测试"),
            ("s_wpe_10", "测试语"),
            ("s_wpe_40", "测式语音"),
            ("m_wpe_10", "测试语音"),
        )
    ]

    comparisons = _paired_deltas(
        results,
        rt60_seconds=(0.6,),
        frontends=("raw", "s_wpe_10", "s_wpe_40", "m_wpe_10"),
        draws=100,
        seed=7,
    )
    pairs = {(row["baseline"], row["candidate"]) for row in comparisons}

    assert ("raw", "m_wpe_10") in pairs
    assert ("s_wpe_10", "m_wpe_10") in pairs
    assert ("s_wpe_40", "m_wpe_10") in pairs
