from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "robust_asr"
    / "train_whisper_lora.py"
)


def _module():
    specification = importlib.util.spec_from_file_location("train_whisper_lora", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _summary() -> dict:
    return {
        "model_id": "openai/whisper-small",
        "model_revision": "a" * 40,
        "utterance_limit": 1000,
        "frontends": ["raw"],
        "rt60_seconds": [0.2, 0.4, 0.6, 0.8, 1.0],
        "conditions": [
            {
                "frontend": "clean",
                "target_rt60_seconds": None,
                "cer": 0.12,
            }
        ],
    }


def test_load_w0_dev_baseline_accepts_exact_frozen_identity(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary()), encoding="utf-8")

    cer, summary = module.load_w0_dev_baseline(
        path,
        model_id="openai/whisper-small",
        model_revision="a" * 40,
        expected_utterances=1000,
        expected_rt60=(0.2, 0.4, 0.6, 0.8, 1.0),
    )

    assert cer == pytest.approx(0.12)
    assert summary["frontends"] == ["raw"]


def test_load_w0_dev_baseline_rejects_wrong_dev_size(tmp_path: Path) -> None:
    module = _module()
    summary = _summary()
    summary["utterance_limit"] = 500
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="dev_model size"):
        module.load_w0_dev_baseline(
            path,
            model_id="openai/whisper-small",
            model_revision="a" * 40,
            expected_utterances=1000,
            expected_rt60=(0.2, 0.4, 0.6, 0.8, 1.0),
        )
