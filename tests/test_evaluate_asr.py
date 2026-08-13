from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from speech_frontend.audio import AudioData, write_audio


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_asr.py"
SPEC = importlib.util.spec_from_file_location("evaluate_asr", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeRuntime:
    model_sha256 = "f" * 64

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, samples: Any, options: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        assert options["temperature"] == 0.0
        assert options["beam_size"] == 5
        return {
            "text": "HELLO, world!",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": len(samples) / 16_000,
                    "text": "HELLO, world!",
                    "temperature": 0.0,
                    "avg_logprob": -0.2,
                    "compression_ratio": 1.1,
                    "no_speech_prob": 0.01,
                }
            ],
        }

    def synchronize(self) -> None:
        pass

    def environment(self) -> dict[str, Any]:
        return {"device": "fake", "fp16": False}


def _config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": MODULE.FROZEN_PROTOCOL_VERSION,
        "implementation": {
            "distribution": "openai-whisper",
            "version": "20250625",
            "torch_version": "2.13.0",
        },
        "model": {
            "name": "small.en",
            "sample_rate_hz": 16_000,
            "sha256": MODULE.FROZEN_MODEL_SHA256,
        },
        "decoding": {
            "task": "transcribe",
            "language": "en",
            "temperature": 0.0,
            "beam_size": 5,
            "patience": 1.0,
            "best_of": None,
            "length_penalty": None,
            "initial_prompt": None,
            "condition_on_previous_text": False,
            "fp16": False,
            "verbose": None,
        },
        "thresholds": {
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        },
        "normalization": {
            "implementation": "whisper.normalizers.EnglishTextNormalizer"
        },
        "timing": {
            "warmup": True,
            "exclude_audio_read": True,
            "exclude_model_load": True,
            "exclude_result_write": True,
        },
    }


def _inputs(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audio = root / "audio.wav"
    write_audio(audio, AudioData(np.zeros(1_600, dtype=np.float32), 16_000))
    audio_sha = MODULE.sha256_file(audio)
    rows = []
    for condition in ("clean", "noisy"):
        rows.append(
            {
                "id": "p001_001",
                "condition": condition,
                "audio": str(audio),
                "audio_sha256": audio_sha,
                "num_samples": 1_600,
                "duration_seconds": 0.1,
                "processing_seconds": 0.01 if condition == "clean" else 0.0,
                "speaker_id": "p001",
                "split": "development",
                "noise": "babble",
                "snr_db": 5,
            }
        )
    references = [{"id": "p001_001", "reference_raw": "Hello world."}]
    return rows, references


def test_evaluate_normalizes_both_sides_and_reuses_cache(tmp_path: Path) -> None:
    inputs, references = _inputs(tmp_path)
    runtime = FakeRuntime()
    kwargs = {
        "input_rows": inputs,
        "reference_rows": references,
        "project_root": tmp_path,
        "output_path": tmp_path / "hypotheses.jsonl",
        "cache_root": tmp_path / "cache",
        "environment_output": tmp_path / "environment.json",
        "config": _config(),
        "runtime": runtime,
        "normalizer": lambda text: text.lower().replace(",", "").replace("!", "").replace(".", ""),
        "warmup": False,
    }
    first = MODULE.evaluate(**kwargs)
    second = MODULE.evaluate(**kwargs)

    assert runtime.calls == 2
    assert all(row["reference_normalized"] == "hello world" for row in first)
    assert all(row["reference_raw"] == "Hello world." for row in first)
    assert all(row["hypothesis_normalized"] == "hello world" for row in first)
    assert first == second
    environment = json.loads((tmp_path / "environment.json").read_text())
    assert environment["device"] == "fake"
    assert environment["cache_hits"] == 2
    assert environment["transcribed_rows"] == 0
    assert all(row["device"] == "fake" for row in second)


def test_fully_cached_resume_skips_warmup_and_inference(tmp_path: Path) -> None:
    inputs, references = _inputs(tmp_path)
    runtime = FakeRuntime()
    kwargs = {
        "input_rows": inputs,
        "reference_rows": references,
        "project_root": tmp_path,
        "output_path": tmp_path / "hypotheses.jsonl",
        "cache_root": tmp_path / "cache",
        "environment_output": tmp_path / "environment.json",
        "config": _config(),
        "runtime": runtime,
        "normalizer": str.lower,
        "warmup": True,
    }

    MODULE.evaluate(**kwargs)
    assert runtime.calls == 3  # one warm-up plus two cache misses
    MODULE.evaluate(**kwargs)
    assert runtime.calls == 3


def test_changed_audio_hash_invalidates_only_matching_condition(tmp_path: Path) -> None:
    inputs, references = _inputs(tmp_path)
    runtime = FakeRuntime()
    common = dict(
        input_rows=inputs,
        reference_rows=references,
        project_root=tmp_path,
        output_path=tmp_path / "hypotheses.jsonl",
        cache_root=tmp_path / "cache",
        environment_output=tmp_path / "environment.json",
        config=_config(),
        runtime=runtime,
        normalizer=str.lower,
        warmup=False,
    )
    MODULE.evaluate(**common)
    inputs[0]["audio_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        MODULE.evaluate(**common)


def test_cache_hit_refreshes_frontend_metadata_and_end_to_end_rtf(
    tmp_path: Path,
) -> None:
    inputs, references = _inputs(tmp_path)
    runtime = FakeRuntime()
    common = dict(
        input_rows=inputs,
        reference_rows=references,
        project_root=tmp_path,
        output_path=tmp_path / "hypotheses.jsonl",
        cache_root=tmp_path / "cache",
        environment_output=tmp_path / "environment.json",
        config=_config(),
        runtime=runtime,
        normalizer=str.lower,
        warmup=False,
    )
    first = MODULE.evaluate(**common)
    inputs[0]["processing_seconds"] = 9.0
    inputs[0]["enhancement_config_digest"] = "changed-provenance"
    second = MODULE.evaluate(**common)

    assert runtime.calls == 2
    assert second[0]["enhancement_config_digest"] == "changed-provenance"
    assert second[0]["end_to_end_seconds"] == pytest.approx(
        first[0]["asr_seconds"] + 9.0
    )


def test_config_rejects_temperature_fallback_shape() -> None:
    config = _config()
    config["decoding"]["temperature"] = [0.0, 0.2]
    with pytest.raises(ValueError, match="temperature"):
        MODULE.validate_config(config)


def test_config_rejects_changed_frozen_threshold() -> None:
    config = _config()
    config["thresholds"]["no_speech_threshold"] = 0.5
    with pytest.raises(ValueError, match="thresholds"):
        MODULE.validate_config(config)


def test_segment_validation_rejects_nonzero_temperature() -> None:
    with pytest.raises(RuntimeError, match="non-zero"):
        MODULE._validate_segments(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "temperature": 0.2,
                    "avg_logprob": -0.2,
                    "compression_ratio": 1.1,
                    "no_speech_prob": 0.01,
                }
            ]
        )


def test_unpaired_conditions_are_rejected(tmp_path: Path) -> None:
    inputs, references = _inputs(tmp_path)
    inputs.append({**inputs[0], "id": "p001_002", "condition": "clean"})
    references.append({"id": "p001_002", "reference_raw": "More words"})
    with pytest.raises(ValueError, match="paired ID set"):
        MODULE._validate_inputs(inputs, MODULE._index_references(references))


def test_asr_inputs_cannot_inject_cached_result_fields(tmp_path: Path) -> None:
    inputs, references = _inputs(tmp_path)
    inputs[0]["hypothesis_normalized"] = "injected"
    with pytest.raises(ValueError, match="reserved result fields"):
        MODULE._validate_inputs(inputs, MODULE._index_references(references))
