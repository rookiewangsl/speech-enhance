from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from speech_frontend.audio import AudioData, write_audio


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_asr_robust.py"
SPEC = importlib.util.spec_from_file_location("evaluate_asr_robust", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeRuntime:
    def __init__(self, model_sha256: str, *, retry_passes: bool = True) -> None:
        self.model_sha256 = model_sha256
        self.retry_passes = retry_passes
        self.calls: list[dict[str, Any]] = []
        self.seeds: list[int] = []

    def set_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def synchronize(self) -> None:
        pass

    def transcribe(self, samples: Any, options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(options))
        temperature = float(options["temperature"])
        if temperature == 0.0:
            text, compression = "", 0.1
        elif self.retry_passes:
            text, compression = "reference words", 0.9
        else:
            text, compression = "repeat repeat repeat repeat " * 20, 10.0
        return {
            "text": text,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": len(samples) / 16_000,
                    "text": text,
                    "temperature": temperature,
                    "avg_logprob": -0.2,
                    "compression_ratio": compression,
                    "no_speech_prob": 0.1,
                }
            ],
        }


def _configs() -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(__file__).parents[1]
    base = json.loads((root / "configs/asr_whisper_small_en.json").read_text())
    robust = json.loads((root / "configs/asr_whisper_small_en_robust_v2.json").read_text())
    robust["retry"]["warmup"] = False
    return base, robust


def _rows(root: Path, base: dict[str, Any], *, anomalous_condition: str = "rnnoise_r3") -> list[dict[str, Any]]:
    audio = root / "audio.wav"
    write_audio(audio, AudioData(np.zeros(16_000, dtype=np.float32), 16_000))
    audio_sha = MODULE.sha256_file(audio)
    reference = "reference words"
    rows: list[dict[str, Any]] = []
    for condition in ("clean", "noisy", "mcra_dd_wiener", "rnnoise_r3"):
        anomalous = condition == anomalous_condition
        text = "this is not a club " * 20 if anomalous else "reference words"
        rows.append(
            {
                "schema_version": 1,
                "id": "p001_001",
                "condition": condition,
                "status": "completed",
                "device": "cpu",
                "model_sha256": base["model"]["sha256"],
                "asr_config_digest": MODULE.canonical_digest(base),
                "evaluator_code_sha256": "e" * 64,
                "runtime_identity_digest": "r" * 64,
                "audio": str(audio),
                "audio_sha256": audio_sha,
                "num_samples": 16_000,
                "duration_seconds": 1.0,
                "processing_seconds": 0.05 if condition in {"mcra_dd_wiener", "rnnoise_r3"} else 0.0,
                "reference_raw": reference,
                "reference_raw_sha256": __import__("hashlib").sha256(reference.encode()).hexdigest(),
                "reference_normalized": reference,
                "hypothesis_raw": text,
                "hypothesis_normalized": text.strip(),
                "asr_seconds": 0.5,
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "text": text,
                        "temperature": 0.0,
                        "avg_logprob": -0.2,
                        "compression_ratio": 10.0 if anomalous else 0.9,
                        "no_speech_prob": 0.1,
                    }
                ],
                "speaker_id": "p001",
                "split": "development",
                "noise": "babble",
                "snr_db": 5.0,
            }
        )
    return rows


def _run(tmp_path: Path, runtime: FakeRuntime, rows: list[dict[str, Any]], base: dict[str, Any], robust: dict[str, Any]) -> list[dict[str, Any]]:
    return MODULE.evaluate_robust(
        v1_rows=rows,
        v1_hypotheses_sha256="h" * 64,
        config=robust,
        base_config=base,
        project_root=tmp_path,
        output_path=tmp_path / "robust.jsonl",
        cache_root=tmp_path / "cache",
        environment_output=tmp_path / "environment.json",
        retry_runtime_identity_value={"device": "cpu", "model_sha256": base["model"]["sha256"]},
        runtime_factory=lambda: runtime,
        normalizer=lambda text: text.lower().strip(),
        token_counter=lambda text: len(str(text).split()),
    )


def test_only_anomalous_row_is_retried_and_first_passing_retry_is_used(tmp_path: Path) -> None:
    base, robust = _configs()
    rows = _rows(tmp_path, base)
    runtime = FakeRuntime(base["model"]["sha256"])

    results = _run(tmp_path, runtime, rows, base, robust)
    rnnoise = next(row for row in results if row["condition"] == "rnnoise_r3")

    assert rnnoise["final_source"] == "temperature_retry"
    assert rnnoise["hypothesis_normalized"] == "reference words"
    assert rnnoise["first_pass_anomalous"] is True
    assert len(rnnoise["retry_attempts"]) == 1
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["temperature"] == 0.2
    assert "beam_size" not in runtime.calls[0]
    assert runtime.calls[0]["best_of"] == 5


def test_failed_retries_use_noisy_and_fully_cached_rerun_loads_no_model(tmp_path: Path) -> None:
    base, robust = _configs()
    rows = _rows(tmp_path, base)
    first_runtime = FakeRuntime(base["model"]["sha256"], retry_passes=False)
    results = _run(tmp_path, first_runtime, rows, base, robust)
    rnnoise = next(row for row in results if row["condition"] == "rnnoise_r3")

    assert rnnoise["final_source"] == "noisy_fallback"
    assert rnnoise["hypothesis_normalized"] == "reference words"
    assert len(rnnoise["retry_attempts"]) == 3

    second_runtime = FakeRuntime(base["model"]["sha256"])
    cached = _run(tmp_path, second_runtime, rows, base, robust)
    assert second_runtime.calls == []
    assert [row["final_source"] for row in cached] == [row["final_source"] for row in results]
    environment = json.loads((tmp_path / "environment.json").read_text())
    assert environment["final_cache_hits"] == 4
    assert environment["model_loads"] == 0


def test_rejects_cross_condition_normalized_reference_mismatch(tmp_path: Path) -> None:
    base, robust = _configs()
    rows = _rows(tmp_path, base)
    rows[-1]["reference_normalized"] = "different words"

    try:
        _run(tmp_path, FakeRuntime(base["model"]["sha256"]), rows, base, robust)
    except ValueError as error:
        assert "paired normalized reference mismatch" in str(error)
    else:
        raise AssertionError("expected normalized reference mismatch to be rejected")
