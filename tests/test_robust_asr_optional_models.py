from __future__ import annotations

import importlib.util

import pytest

from robust_asr.lora import LoRAProtocol
from robust_asr.models.whisper_lora import (
    _build_whisper_lora_config,
    load_whisper_lora_components,
)


def test_whisper_lora_uses_transparent_peft_wrapper() -> None:
    pytest.importorskip("peft")

    config = _build_whisper_lora_config(
        settings=LoRAProtocol(),
        target_modules=("model.encoder.layers.0.self_attn.q_proj",),
    )

    assert config.task_type is None
    assert config.target_modules == {
        "model.encoder.layers.0.self_attn.q_proj"
    }


def test_whisper_loader_reports_missing_training_dependencies() -> None:
    if importlib.util.find_spec("transformers") is not None and importlib.util.find_spec(
        "peft"
    ) is not None:
        pytest.skip("formal Whisper dependencies are installed")

    with pytest.raises(RuntimeError, match="transformers and peft"):
        load_whisper_lora_components(local_files_only=True)
