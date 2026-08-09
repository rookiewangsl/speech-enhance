"""Improved minima-controlled recursive averaging (IMCRA).

The implementation follows Cohen's two-iteration structure: a rough
time-frequency minimum tracker first excludes strong speech components, then a
second tracker controls a Bayesian speech-presence probability and the noise
PSD update.  It is kept frame-causal and independent from the gain estimator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complexfloating]
FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class IMCRAConfig:
    """Parameters adapted to 16 kHz, 512-point STFT, and 128-sample hops."""

    alpha_s: float = 0.9
    alpha_d: float = 0.85
    subwindow_frames: int = 15
    history_subwindows: int = 8
    minimum_bias: float = 1.66
    noise_bias: float = 1.4685
    rough_power_threshold: float = 4.6
    rough_smoothed_threshold: float = 1.67
    soft_power_threshold: float = 3.0
    alpha_eta: float = 0.95
    minimum_prior_snr_db: float = -18.0
    maximum_snr: float = 1e4
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("alpha_s", "alpha_d", "alpha_eta"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.subwindow_frames <= 0 or self.history_subwindows <= 0:
            raise ValueError("minimum-tracking window sizes must be positive")
        for name in (
            "minimum_bias",
            "noise_bias",
            "rough_power_threshold",
            "rough_smoothed_threshold",
            "soft_power_threshold",
            "maximum_snr",
            "epsilon",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.soft_power_threshold <= 1.0:
            raise ValueError("soft_power_threshold must exceed one")


class IMCRA:
    """Causal two-iteration noise PSD estimator for nonstationary noise."""

    def __init__(self, config: IMCRAConfig | None = None) -> None:
        self.config = config or IMCRAConfig()
        self.reset()

    def reset(self) -> None:
        """Clear all utterance-local smoothing and minimum-tracking state."""

        self._frame_count = 0
        self._first_smoothed: NDArray[np.float64] | None = None
        self._second_smoothed: NDArray[np.float64] | None = None
        self._noise_average: NDArray[np.float64] | None = None
        self._noise_psd: NDArray[np.float64] | None = None
        self._first_active_minimum: NDArray[np.float64] | None = None
        self._second_active_minimum: NDArray[np.float64] | None = None
        self._first_history: deque[NDArray[np.float64]] = deque(
            maxlen=self.config.history_subwindows
        )
        self._second_history: deque[NDArray[np.float64]] = deque(
            maxlen=self.config.history_subwindows
        )

    def process_frame(
        self,
        noisy_spectrum: ComplexArray,
        *,
        previous_decision_directed_term: FloatArray | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update and return ``(noise_psd, speech_presence_probability)``."""

        spectrum = np.asarray(noisy_spectrum)
        if spectrum.ndim != 1 or spectrum.size == 0:
            raise ValueError("one non-empty spectrum is required")
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("spectrum contains NaN or infinite values")
        power = np.abs(spectrum) ** 2
        frequency_smoothed = self._frequency_smooth(power)

        if self._noise_psd is None:
            self._initialize(power, frequency_smoothed)
            return self._noise_psd.copy(), np.zeros_like(power)
        self._require_matching_bins(spectrum)

        if previous_decision_directed_term is None:
            previous_term = np.zeros_like(power)
        else:
            previous_term = np.asarray(
                previous_decision_directed_term,
                dtype=np.float64,
            )
            if previous_term.shape != power.shape:
                raise ValueError("decision-directed term must match spectrum")
            if not np.all(np.isfinite(previous_term)) or np.any(previous_term < 0.0):
                raise ValueError("decision-directed term must be finite and non-negative")

        posterior = np.clip(
            power / np.maximum(self._noise_psd, self.config.epsilon),
            0.0,
            self.config.maximum_snr,
        )
        prior = np.clip(
            self.config.alpha_eta * previous_term
            + (1.0 - self.config.alpha_eta) * np.maximum(posterior - 1.0, 0.0),
            10.0 ** (self.config.minimum_prior_snr_db / 10.0),
            self.config.maximum_snr,
        )
        v = posterior * prior / (1.0 + prior)

        self._first_smoothed = (
            self.config.alpha_s * self._first_smoothed
            + (1.0 - self.config.alpha_s) * frequency_smoothed
        )
        first_minimum = self._current_minimum(
            self._first_history,
            self._first_active_minimum,
        )
        rough_noise_mask = (
            (power < self.config.rough_power_threshold * self.config.minimum_bias * first_minimum)
            & (
                self._first_smoothed
                < self.config.rough_smoothed_threshold
                * self.config.minimum_bias
                * first_minimum
            )
        )
        second_input = self._masked_frequency_smooth(
            power,
            rough_noise_mask,
            fallback=self._second_smoothed,
        )
        self._second_smoothed = (
            self.config.alpha_s * self._second_smoothed
            + (1.0 - self.config.alpha_s) * second_input
        )
        self._update_minimum_trackers()
        second_minimum = self._current_minimum(
            self._second_history,
            self._second_active_minimum,
        )

        normalized_power = power / np.maximum(
            self.config.minimum_bias * second_minimum,
            self.config.epsilon,
        )
        normalized_smoothed = self._first_smoothed / np.maximum(
            self.config.minimum_bias * second_minimum,
            self.config.epsilon,
        )
        absence_prior = np.ones_like(power)
        transition = (
            (normalized_power > 1.0)
            & (normalized_power < self.config.soft_power_threshold)
            & (normalized_smoothed < self.config.rough_smoothed_threshold)
        )
        absence_prior[transition] = (
            self.config.soft_power_threshold - normalized_power[transition]
        ) / (self.config.soft_power_threshold - 1.0)
        strong_presence = (
            (normalized_power >= self.config.soft_power_threshold)
            | (normalized_smoothed >= self.config.rough_smoothed_threshold)
        )
        absence_prior[strong_presence] = 0.0

        speech_probability = np.zeros_like(power)
        soft = (absence_prior > 0.0) & (absence_prior < 1.0)
        odds = absence_prior[soft] / np.maximum(
            1.0 - absence_prior[soft],
            self.config.epsilon,
        )
        speech_probability[soft] = 1.0 / (
            1.0 + odds * (1.0 + prior[soft]) * np.exp(-v[soft])
        )
        speech_probability[absence_prior <= 0.0] = 1.0

        smoothing = self.config.alpha_d + (
            1.0 - self.config.alpha_d
        ) * speech_probability
        self._noise_average = (
            smoothing * self._noise_average + (1.0 - smoothing) * power
        )
        self._noise_psd = self.config.noise_bias * self._noise_average
        self._frame_count += 1
        return self._noise_psd.copy(), speech_probability.copy()

    def _initialize(
        self,
        power: NDArray[np.float64],
        frequency_smoothed: NDArray[np.float64],
    ) -> None:
        self._first_smoothed = frequency_smoothed.copy()
        self._second_smoothed = frequency_smoothed.copy()
        self._noise_average = power.copy()
        self._noise_psd = self.config.noise_bias * power
        self._first_active_minimum = frequency_smoothed.copy()
        self._second_active_minimum = frequency_smoothed.copy()
        self._frame_count = 1

    def _update_minimum_trackers(self) -> None:
        self._first_active_minimum = np.minimum(
            self._first_active_minimum,
            self._first_smoothed,
        )
        self._second_active_minimum = np.minimum(
            self._second_active_minimum,
            self._second_smoothed,
        )
        if self._frame_count % self.config.subwindow_frames != 0:
            return
        self._first_history.append(self._first_active_minimum.copy())
        self._second_history.append(self._second_active_minimum.copy())
        self._first_active_minimum = self._first_smoothed.copy()
        self._second_active_minimum = self._second_smoothed.copy()

    @staticmethod
    def _current_minimum(
        history: deque[NDArray[np.float64]],
        active: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if not history:
            return active
        return np.minimum(active, np.minimum.reduce(history))

    @staticmethod
    def _frequency_smooth(power: NDArray[np.float64]) -> NDArray[np.float64]:
        padded = np.pad(power, (1, 1), mode="edge")
        return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]

    @staticmethod
    def _masked_frequency_smooth(
        power: NDArray[np.float64],
        mask: NDArray[np.bool_],
        *,
        fallback: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        weights = np.array([0.25, 0.5, 0.25])
        padded_power = np.pad(power * mask, (1, 1), mode="edge")
        padded_mask = np.pad(mask.astype(np.float64), (1, 1), mode="edge")
        numerator = (
            weights[0] * padded_power[:-2]
            + weights[1] * padded_power[1:-1]
            + weights[2] * padded_power[2:]
        )
        denominator = (
            weights[0] * padded_mask[:-2]
            + weights[1] * padded_mask[1:-1]
            + weights[2] * padded_mask[2:]
        )
        return np.divide(
            numerator,
            denominator,
            out=fallback.copy(),
            where=denominator > 0.0,
        )

    def _require_matching_bins(self, spectrum: ComplexArray) -> None:
        if spectrum.shape != self._noise_psd.shape:
            raise ValueError("frequency-bin count changed without reset")
