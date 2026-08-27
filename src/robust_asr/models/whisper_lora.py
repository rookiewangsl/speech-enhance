"""Lazy Hugging Face Whisper and PEFT integration.

Importing this module is safe in the lightweight DSP environment. Calling the
loader is intentionally deferred until model storage and training dependencies
are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robust_asr.lora import (
    LoRAProtocol,
    LoRATarget,
    select_lora_target_names,
    validate_whisper_target_count,
)


@dataclass(frozen=True)
class WhisperLoRAComponents:
    processor: Any
    model: Any
    target_module_names: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int


def _build_whisper_lora_config(
    *,
    settings: LoRAProtocol,
    target_modules: tuple[str, ...],
):
    """Build a generic PEFT config that preserves Whisper ``input_features``.

    ``PeftModelForSeq2SeqLM`` assumes a text encoder and injects ``input_ids``
    into the wrapped model. Whisper is a speech conditional-generation model;
    its encoder input is ``input_features``. Leaving ``task_type`` unset makes
    PEFT use its generic transparent wrapper while retaining the same LoRA
    adapters, save/load format, and trainable-parameter semantics.
    """

    try:
        from peft import LoraConfig
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Whisper LoRA training requires peft") from exc
    return LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias=settings.bias,
        task_type=None,
        target_modules=list(target_modules),
    )


def load_whisper_lora_components(
    *,
    model_id: str = "openai/whisper-small",
    target: LoRATarget = LoRATarget.ENCODER_QV,
    protocol: LoRAProtocol | None = None,
    revision: str = "973afd24965f72e36ca33b3055d56a652f456b4d",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> WhisperLoRAComponents:
    """Load Whisper, validate exact target names, and attach LoRA adapters."""

    try:
        from peft import get_peft_model
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as exc:  # pragma: no cover - installed after storage arrives
        raise RuntimeError(
            "Whisper LoRA training requires transformers and peft"
        ) from exc

    settings = protocol or LoRAProtocol()
    load_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
    }
    processor = WhisperProcessor.from_pretrained(model_id, **load_kwargs)
    processor.tokenizer.set_prefix_tokens(
        language="zh",
        task="transcribe",
        predict_timestamps=False,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, **load_kwargs
    )
    model.generation_config.language = "zh"
    model.generation_config.task = "transcribe"
    names = tuple(name for name, _ in model.named_modules())
    targets = select_lora_target_names(names, target)
    encoder_layers = len(model.model.encoder.layers)
    decoder_layers = len(model.model.decoder.layers)
    validate_whisper_target_count(
        targets,
        target,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    )
    peft_config = _build_whisper_lora_config(
        settings=settings,
        target_modules=targets,
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if unexpected:
        raise RuntimeError(
            f"non-LoRA parameters unexpectedly remain trainable: {unexpected[:5]}"
        )
    return WhisperLoRAComponents(
        processor=processor,
        model=model,
        target_module_names=targets,
        trainable_parameters=trainable,
        total_parameters=total,
    )
