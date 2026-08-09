"""Short-time Fourier analysis with perfect-reconstruction synthesis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from speech_frontend.framing import (
    frame_signal,
    sqrt_periodic_hann,
    weighted_overlap_add,
)

FloatArray = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]


@dataclass(frozen=True)
class STFTConfig:
    """STFT analysis and synthesis parameters."""

    frame_length: int = 512
    hop_length: int = 128
    fft_length: int = 512

    def __post_init__(self) -> None:
        if self.frame_length <= 0:
            raise ValueError("frame_length must be positive")
        if self.hop_length <= 0 or self.hop_length >= self.frame_length:
            raise ValueError("hop_length must be in [1, frame_length)")
        if self.fft_length < self.frame_length:
            raise ValueError("fft_length must be at least frame_length")


@dataclass(frozen=True)
class STFTResult:
    """STFT values plus the metadata required for exact synthesis."""

    spectrum: ComplexArray
    original_length: int
    left_padding: int


class STFT:
    """STFT/iSTFT pair using a square-root periodic Hann window."""

    def __init__(self, config: STFTConfig | None = None) -> None:
        self.config = config or STFTConfig()
        self.window = sqrt_periodic_hann(self.config.frame_length)

    def analyze(self, signal: FloatArray) -> STFTResult:
        """Convert a real mono signal into frame-major complex spectra."""

        signal = np.asarray(signal)
        if signal.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        if not np.all(np.isfinite(signal)):
            raise ValueError("signal contains NaN or infinite values")

        frames, left_padding = frame_signal(
            signal,
            self.config.frame_length,
            self.config.hop_length,
        )
        windowed = frames * self.window[np.newaxis, :]
        spectrum = np.fft.rfft(
            windowed,
            n=self.config.fft_length,
            axis=1,
        )
        return STFTResult(
            spectrum=spectrum,
            original_length=signal.size,
            left_padding=left_padding,
        )

    def synthesize(self, result: STFTResult) -> NDArray[np.float64]:
        """Reconstruct a signal from an :class:`STFTResult`."""

        spectrum = np.asarray(result.spectrum)
        if spectrum.ndim != 2:
            raise ValueError("spectrum must be two-dimensional")
        if spectrum.shape[0] == 0:
            raise ValueError("spectrum must contain at least one frame")
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("spectrum contains NaN or infinite values")
        expected_bins = self.config.fft_length // 2 + 1
        if spectrum.shape[1] != expected_bins:
            raise ValueError("spectrum has an unexpected frequency-bin count")
        if result.original_length < 0:
            raise ValueError("original_length cannot be negative")
        expected_left_padding = (
            self.config.frame_length - self.config.hop_length
        )
        if result.left_padding != expected_left_padding:
            raise ValueError(
                "left_padding is inconsistent with the STFT configuration"
            )

        frames = np.fft.irfft(
            spectrum,
            n=self.config.fft_length,
            axis=1,
        )[:, : self.config.frame_length]
        return weighted_overlap_add(
            frames,
            self.window,
            self.config.hop_length,
            output_length=result.original_length,
            left_padding=result.left_padding,
        )
