"""Minima-controlled recursive averaging (MCRA) noise PSD estimation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complexfloating]


@dataclass(frozen=True)
class MCRAConfig:
    """Initial parameters from Cohen and Berdugo's MCRA formulation."""

    alpha_d: float = 0.95
    alpha_p: float = 0.2
    alpha_s: float = 0.8
    minimum_window_frames: int = 125
    ratio_threshold: float = 5.0
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("alpha_d", "alpha_p", "alpha_s"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.minimum_window_frames <= 0:
            raise ValueError("minimum_window_frames must be positive")
        if self.ratio_threshold <= 1.0:
            raise ValueError("ratio_threshold must exceed one")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")


class MCRA:
    """Causal MCRA estimator returning a PSD and speech-presence probability.

    The returned probability is intentionally local to the noise estimator. It
    must not be treated as a final global VAD decision.
    """

    def __init__(self, config: MCRAConfig | None = None) -> None:
        self.config = config or MCRAConfig()
        self.reset()

    def reset(self) -> None:
        """Forget all per-utterance state."""

        self._smoothed_power: NDArray[np.float64] | None = None
        self._noise_psd: NDArray[np.float64] | None = None
        self._speech_probability: NDArray[np.float64] | None = None
        self._minimum_history: deque[NDArray[np.float64]] = deque(
            maxlen=self.config.minimum_window_frames
        )

    def process_frame(
        self,
        noisy_spectrum: ComplexArray,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update state from one spectrum and return ``(noise_psd, p_N)``."""

        spectrum = np.asarray(noisy_spectrum)
        if spectrum.ndim != 1:
            raise ValueError("noisy_spectrum must be one-dimensional")
        if spectrum.size == 0:
            raise ValueError("noisy_spectrum cannot be empty")
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("noisy_spectrum contains NaN or infinite values")

        power = np.abs(spectrum) ** 2
        frequency_smoothed = self._frequency_smooth(power)
        if self._smoothed_power is None:
            self._smoothed_power = frequency_smoothed
            self._noise_psd = power.copy()
            self._speech_probability = np.zeros_like(power)
        else:
            self._require_matching_bins(spectrum)
            self._smoothed_power = (
                self.config.alpha_s * self._smoothed_power
                + (1.0 - self.config.alpha_s) * frequency_smoothed
            )

        self._minimum_history.append(self._smoothed_power.copy())
        local_minimum = np.minimum.reduce(self._minimum_history)
        ratio = self._smoothed_power / np.maximum(
            local_minimum,
            self.config.epsilon,
        )
        indicator = (ratio > self.config.ratio_threshold).astype(np.float64)
        self._speech_probability = (
            self.config.alpha_p * self._speech_probability
            + (1.0 - self.config.alpha_p) * indicator
        )
        smoothing = self.config.alpha_d + (
            1.0 - self.config.alpha_d
        ) * self._speech_probability
        self._noise_psd = (
            smoothing * self._noise_psd + (1.0 - smoothing) * power
        )
        return self._noise_psd.copy(), self._speech_probability.copy()

    def process_spectra(
        self,
        spectra: ComplexArray,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Process a frame-major spectrum matrix after resetting the state."""

        values = np.asarray(spectra)
        if values.ndim != 2:
            raise ValueError("spectra must be frame-major and two-dimensional")
        if values.shape[0] == 0:
            raise ValueError("spectra must contain at least one frame")
        self.reset()
        noise: list[NDArray[np.float64]] = []
        probability: list[NDArray[np.float64]] = []
        for spectrum in values:
            psd, p_n = self.process_frame(spectrum)
            noise.append(psd)
            probability.append(p_n)
        return np.stack(noise), np.stack(probability)

    def _require_matching_bins(self, spectrum: ComplexArray) -> None:
        if spectrum.shape != self._smoothed_power.shape:
            raise ValueError("frequency-bin count changed without reset")

    @staticmethod
    def _frequency_smooth(
        power: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        padded = np.pad(power, (1, 1), mode="edge")
        return (
            0.25 * padded[:-2]
            + 0.5 * padded[1:-1]
            + 0.25 * padded[2:]
        )
