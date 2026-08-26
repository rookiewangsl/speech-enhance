from __future__ import annotations

import importlib.util

import pytest

from robust_asr.models.whisper_lora import load_whisper_lora_components


def test_whisper_loader_reports_missing_training_dependencies() -> None:
    if importlib.util.find_spec("transformers") is not None and importlib.util.find_spec(
        "peft"
    ) is not None:
        pytest.skip("formal Whisper dependencies are installed")

    with pytest.raises(RuntimeError, match="transformers and peft"):
        load_whisper_lora_components(local_files_only=True)

