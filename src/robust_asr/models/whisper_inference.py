"""Frozen multilingual Whisper inference with no training dependencies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class FrozenWhisper:
    """Small inference-only wrapper shared by smoke and formal baselines."""

    def __init__(
        self,
        model_id: str = "openai/whisper-small",
        *,
        revision: str = "973afd24965f72e36ca33b3055d56a652f456b4d",
        cache_dir: str | Path | None = None,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, WhisperForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "frozen Whisper requires torch and transformers"
            ) from exc
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"unsupported device: {device}")
        self.model_id = model_id
        self.model_revision = revision
        self.device = device
        self._torch = torch
        load = {
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "local_files_only": local_files_only,
            "revision": revision,
        }
        self.processor = AutoProcessor.from_pretrained(model_id, **load)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, **load)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)

    def transcribe(
        self,
        audio: NDArray[np.floating],
        *,
        sample_rate: int = 16_000,
    ) -> str:
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Whisper input must be non-empty mono audio")
        if sample_rate != 16_000:
            raise ValueError("Whisper baseline requires 16 kHz audio")
        if waveform.size > 30 * sample_rate:
            raise ValueError(
                "Whisper baseline input exceeds 30 seconds; refusing silent truncation"
            )
        if not np.all(np.isfinite(waveform)):
            raise ValueError("Whisper input contains NaN or infinite values")
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(self.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        generation = {
            "input_features": input_features,
            "language": "zh",
            "task": "transcribe",
            "do_sample": False,
            "num_beams": 1,
            "max_length": 225,
        }
        if attention_mask is not None:
            generation["attention_mask"] = attention_mask
        with self._torch.inference_mode():
            token_ids = self.model.generate(**generation)
        return str(
            self.processor.batch_decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        ).strip()
