"""Frozen Paraformer inference used only for cross-model validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


DEFAULT_MODEL_ID = (
    "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)
DEFAULT_MODEL_REVISION = "v2.0.4"
DEFAULT_MODEL_SHA256 = (
    "5bba782a5e9196166233b9ab12ba04cadff9ef9212b4ff6153ed9290ff679025"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_paraformer_text(result: Any) -> str:
    """Validate the one-utterance FunASR result and return its text."""

    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise RuntimeError("Paraformer returned a non-sequence result")
    if len(result) != 1 or not isinstance(result[0], Mapping):
        raise RuntimeError("Paraformer must return exactly one result per utterance")
    text = result[0].get("text")
    if not isinstance(text, str):
        raise RuntimeError("Paraformer result does not contain string field 'text'")
    return text.strip()


class FrozenParaformer:
    """Classic offline Paraformer without VAD, punctuation, LM or hotwords."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_MODEL_REVISION,
        expected_model_sha256: str = DEFAULT_MODEL_SHA256,
        device: str = "auto",
        cpu_threads: int = 32,
    ) -> None:
        try:
            import torch
            from funasr import AutoModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "frozen Paraformer requires torch, torchaudio and funasr"
            ) from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"unsupported Paraformer device: {device}")
        if cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive")

        local_path = Path(model_path).expanduser().resolve()
        required = ("model.pt", "config.yaml", "am.mvn", "tokens.json")
        missing = [name for name in required if not (local_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"incomplete Paraformer snapshot {local_path}: missing {missing}"
            )
        observed_sha256 = _sha256(local_path / "model.pt")
        if observed_sha256 != expected_model_sha256:
            raise ValueError(
                "Paraformer model.pt SHA-256 mismatch: "
                f"{observed_sha256} != {expected_model_sha256}"
            )

        self.model_id = model_id
        self.model_revision = (
            f"{revision}+model.pt.sha256.{observed_sha256}"
        )
        self.device = device
        self.model_path = str(local_path)
        self.model_artifact_sha256 = observed_sha256
        self._model = AutoModel(
            model=str(local_path),
            model_revision=revision,
            hub="ms",
            device=device,
            ncpu=cpu_threads,
            disable_update=True,
            disable_pbar=True,
            log_level="WARNING",
        )

    def transcribe(
        self,
        audio: NDArray[np.floating],
        *,
        sample_rate: int = 16_000,
    ) -> str:
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Paraformer input must be non-empty mono audio")
        if sample_rate != 16_000:
            raise ValueError("Paraformer cross-check requires 16 kHz audio")
        if not np.all(np.isfinite(waveform)):
            raise ValueError("Paraformer input contains NaN or infinite values")
        result = self._model.generate(
            input=waveform,
            batch_size_s=300,
            disable_pbar=True,
        )
        return _extract_paraformer_text(result)
