"""External industrial VAD baselines, kept separate from project core logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from speech_frontend.vad.energy import VADResult


@dataclass(frozen=True)
class WebRTCVADConfig:
    """WebRTC VAD API geometry and aggressiveness setting."""

    sample_rate: int = 16_000
    frame_ms: int = 20
    hop_ms: int = 10
    aggressiveness: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate not in (8_000, 16_000, 32_000, 48_000):
            raise ValueError("WebRTC VAD supports 8, 16, 32, or 48 kHz")
        if self.frame_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD frame_ms must be 10, 20, or 30")
        if self.hop_ms <= 0 or self.hop_ms > self.frame_ms:
            raise ValueError("hop_ms must lie in (0, frame_ms]")
        if self.aggressiveness not in (0, 1, 2, 3):
            raise ValueError("aggressiveness must be an integer from 0 to 3")


class WebRTCVAD:
    """Thin wrapper around WebRTC VAD for a fair external comparison."""

    def __init__(self, config: WebRTCVADConfig | None = None) -> None:
        self.config = config or WebRTCVADConfig()
        try:
            import webrtcvad
        except ImportError as error:
            raise RuntimeError(
                "WebRTC VAD is optional; install the 'external' project extra"
            ) from error
        self._vad = webrtcvad.Vad(self.config.aggressiveness)

    def detect(self, samples: NDArray[np.floating]) -> VADResult:
        """Return one binary decision per overlapping frame."""

        signal = np.asarray(samples)
        if signal.ndim != 1 or not np.all(np.isfinite(signal)):
            raise ValueError("samples must be a finite one-dimensional array")
        frame_length = self.config.frame_ms * self.config.sample_rate // 1_000
        hop_length = self.config.hop_ms * self.config.sample_rate // 1_000
        frame_count = (
            1
            if signal.size <= frame_length
            else int(np.ceil((signal.size - frame_length) / hop_length)) + 1
        )
        padded_length = (frame_count - 1) * hop_length + frame_length
        padded = np.pad(signal, (0, padded_length - signal.size))
        pcm = np.round(np.clip(padded, -1.0, 1.0) * 32_767.0).astype("<i2")
        labels = np.empty(frame_count, dtype=np.bool_)
        for index in range(frame_count):
            start = index * hop_length
            frame = pcm[start : start + frame_length]
            labels[index] = self._vad.is_speech(
                frame.tobytes(),
                self.config.sample_rate,
            )
        padded_labels = np.pad(labels.astype(np.int8), (1, 1))
        changes = np.diff(padded_labels)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        segments = tuple(
            (
                int(start * hop_length),
                min(
                    signal.size,
                    int((stop - 1) * hop_length + frame_length),
                ),
            )
            for start, stop in zip(starts, stops, strict=True)
        )
        binary = labels.astype(np.float64)
        return VADResult(
            speech_probability=binary,
            frame_labels=labels,
            energy_db=binary,
            noise_floor_db=np.zeros(frame_count),
            snr_margin_db=binary,
            segments=segments,
        )
