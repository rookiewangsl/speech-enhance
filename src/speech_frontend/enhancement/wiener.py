"""Instantaneous and decision-directed Wiener gains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complexfloating]
FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class WienerConfig:
    """Numerical and smoothing settings for DD-Wiener processing."""

    alpha_dd: float = 0.96
    gain_floor: float = 0.30
    gain_decrease_smoothing: float = 0.7
    gain_increase_smoothing: float = 0.4
    gain_frequency_smoothing: float = 0.0
    startup_frames: int = 0
    speech_absence_prior_min: float = 0.05
    speech_absence_prior_max: float = 0.95
    vad_prior_strength: float = 1.0
    maximum_snr: float = 1e4
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "alpha_dd",
            "gain_decrease_smoothing",
            "gain_increase_smoothing",
            "gain_frequency_smoothing",
        ):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0.0 <= self.gain_floor <= 1.0:
            raise ValueError("gain_floor must be in [0, 1]")
        if self.startup_frames < 0:
            raise ValueError("startup_frames must be non-negative")
        if not (
            0.0
            < self.speech_absence_prior_min
            < self.speech_absence_prior_max
            < 1.0
        ):
            raise ValueError("speech absence prior bounds must lie in (0, 1)")
        if not 0.0 <= self.vad_prior_strength <= 1.0:
            raise ValueError("vad_prior_strength must be in [0, 1]")
        if self.maximum_snr <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("SNR limit and epsilon must be positive")


def instantaneous_wiener_gain(
    noisy_power: FloatArray,
    noise_psd: FloatArray,
    *,
    gain_floor: float = 0.05,
    maximum_snr: float = 1e4,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Compute a posteriori subtraction followed by a Wiener gain."""

    power = np.asarray(noisy_power, dtype=np.float64)
    noise = np.asarray(noise_psd, dtype=np.float64)
    if power.shape != noise.shape:
        raise ValueError("noisy_power and noise_psd shapes must match")
    if (
        not np.all(np.isfinite(power))
        or not np.all(np.isfinite(noise))
        or np.any(power < 0.0)
        or np.any(noise < 0.0)
    ):
        raise ValueError("power and PSD must be finite and non-negative")
    if not 0.0 <= gain_floor <= 1.0:
        raise ValueError("gain_floor must be in [0, 1]")

    posterior_snr = np.clip(
        power / np.maximum(noise, epsilon),
        0.0,
        maximum_snr,
    )
    prior_snr = np.maximum(posterior_snr - 1.0, 0.0)
    gain = prior_snr / (1.0 + prior_snr)
    return np.clip(gain, gain_floor, 1.0)


class DecisionDirectedWiener:
    """Stateful frame processor with asymmetric gain smoothing."""

    def __init__(self, config: WienerConfig | None = None) -> None:
        self.config = config or WienerConfig()
        self.reset()

    def reset(self) -> None:
        """Clear state so no information leaks between utterances."""

        self._previous_gain: NDArray[np.float64] | None = None
        self._previous_posterior_snr: NDArray[np.float64] | None = None
        self._processed_frames = 0

    def process_frame(
        self,
        noisy_spectrum: ComplexArray,
        noise_psd: FloatArray,
    ) -> tuple[ComplexArray, NDArray[np.float64]]:
        """Enhance one spectrum and return it with the applied gain."""

        spectrum = np.asarray(noisy_spectrum)
        noise = np.asarray(noise_psd, dtype=np.float64)
        if spectrum.ndim != 1 or noise.shape != spectrum.shape:
            raise ValueError("one spectrum and matching noise PSD are required")
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("spectrum contains NaN or infinite values")
        if not np.all(np.isfinite(noise)) or np.any(noise < 0.0):
            raise ValueError("noise_psd must be finite and non-negative")

        posterior = np.clip(
            np.abs(spectrum) ** 2
            / np.maximum(noise, self.config.epsilon),
            0.0,
            self.config.maximum_snr,
        )
        instantaneous_prior = np.maximum(posterior - 1.0, 0.0)
        if self._previous_gain is None:
            prior = instantaneous_prior
            previous_gain = np.ones_like(prior)
        else:
            if self._previous_gain.shape != spectrum.shape:
                raise ValueError("frequency-bin count changed without reset")
            prior = (
                self.config.alpha_dd
                * self._previous_gain**2
                * self._previous_posterior_snr
                + (1.0 - self.config.alpha_dd) * instantaneous_prior
            )
            previous_gain = self._previous_gain
        prior = np.clip(prior, 0.0, self.config.maximum_snr)
        raw_gain = np.clip(
            prior / (1.0 + prior),
            self.config.gain_floor,
            1.0,
        )

        alpha = np.where(
            raw_gain < previous_gain,
            self.config.gain_decrease_smoothing,
            self.config.gain_increase_smoothing,
        )
        gain = alpha * previous_gain + (1.0 - alpha) * raw_gain
        gain = _artifact_aware_gain_smoothing(
            gain,
            frequency_smoothing=self.config.gain_frequency_smoothing,
            processed_frames=self._processed_frames,
            startup_frames=self.config.startup_frames,
        )
        gain = np.clip(gain, self.config.gain_floor, 1.0)
        self._previous_gain = gain
        self._previous_posterior_snr = posterior
        self._processed_frames += 1
        return gain * spectrum, gain.copy()


class DualUncertaintyWiener:
    """DD-Wiener with separate MCRA and global-VAD uncertainty roles.

    MCRA supplies the input ``noise_psd`` elsewhere.  This class consumes a
    global VAD probability only to form the gain-side speech-presence
    probability, never to hard-gate the noise estimator.
    """

    def __init__(self, config: WienerConfig | None = None) -> None:
        self.config = config or WienerConfig()
        self.reset()

    def reset(self) -> None:
        self._previous_gain: NDArray[np.float64] | None = None
        self._previous_posterior_snr: NDArray[np.float64] | None = None
        self._processed_frames = 0

    def process_frame(
        self,
        noisy_spectrum: ComplexArray,
        noise_psd: FloatArray,
        *,
        vad_speech_probability: float,
    ) -> tuple[ComplexArray, NDArray[np.float64], NDArray[np.float64]]:
        """Return enhanced spectrum, soft gain, and gain-side probability."""

        spectrum = np.asarray(noisy_spectrum)
        noise = np.asarray(noise_psd, dtype=np.float64)
        if spectrum.ndim != 1 or noise.shape != spectrum.shape:
            raise ValueError("one spectrum and matching noise PSD are required")
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("spectrum contains NaN or infinite values")
        if not np.all(np.isfinite(noise)) or np.any(noise < 0.0):
            raise ValueError("noise_psd must be finite and non-negative")
        if not np.isfinite(vad_speech_probability):
            raise ValueError("vad_speech_probability must be finite")

        posterior = np.clip(
            np.abs(spectrum) ** 2
            / np.maximum(noise, self.config.epsilon),
            0.0,
            self.config.maximum_snr,
        )
        instantaneous_prior = np.maximum(posterior - 1.0, 0.0)
        if self._previous_gain is None:
            prior = instantaneous_prior
            previous_gain = np.ones_like(prior)
        else:
            if self._previous_gain.shape != spectrum.shape:
                raise ValueError("frequency-bin count changed without reset")
            prior = (
                self.config.alpha_dd
                * self._previous_gain**2
                * self._previous_posterior_snr
                + (1.0 - self.config.alpha_dd) * instantaneous_prior
            )
            previous_gain = self._previous_gain
        prior = np.clip(prior, 0.0, self.config.maximum_snr)
        wiener_gain = prior / (1.0 + prior)
        calibrated_vad_probability = 0.5 + self.config.vad_prior_strength * (
            np.clip(vad_speech_probability, 0.0, 1.0) - 0.5
        )
        speech_absence_prior = np.clip(
            1.0 - calibrated_vad_probability,
            self.config.speech_absence_prior_min,
            self.config.speech_absence_prior_max,
        )
        log_odds = np.log(speech_absence_prior) - np.log1p(
            -speech_absence_prior
        )
        likelihood_term = posterior * prior / (1.0 + prior)
        gain_speech_probability = 1.0 / (
            1.0
            + np.exp(
                np.clip(
                    log_odds + np.log1p(prior) - likelihood_term,
                    -60.0,
                    60.0,
                )
            )
        )
        raw_gain = (
            gain_speech_probability * wiener_gain
            + (1.0 - gain_speech_probability) * self.config.gain_floor
        )
        alpha = np.where(
            raw_gain < previous_gain,
            self.config.gain_decrease_smoothing,
            self.config.gain_increase_smoothing,
        )
        gain = _artifact_aware_gain_smoothing(
            alpha * previous_gain + (1.0 - alpha) * raw_gain,
            frequency_smoothing=self.config.gain_frequency_smoothing,
            processed_frames=self._processed_frames,
            startup_frames=self.config.startup_frames,
        )
        gain = np.clip(gain, self.config.gain_floor, 1.0)
        self._previous_gain = gain
        self._previous_posterior_snr = posterior
        self._processed_frames += 1
        return gain * spectrum, gain.copy(), gain_speech_probability.copy()


def _artifact_aware_gain_smoothing(
    gain: NDArray[np.float64],
    *,
    frequency_smoothing: float,
    processed_frames: int,
    startup_frames: int,
) -> NDArray[np.float64]:
    """Limit isolated time-frequency gain changes that sound metallic.

    A five-bin binomial smoother removes narrow, frame-local gain notches, a
    common source of musical noise.  The optional causal start-up ramp avoids
    applying an unreliable first noise estimate at full strength.
    """

    values = np.asarray(gain, dtype=np.float64)
    if frequency_smoothing > 0.0:
        padded = np.pad(values, (2, 2), mode="edge")
        local_average = (
            padded[:-4]
            + 4.0 * padded[1:-3]
            + 6.0 * padded[2:-2]
            + 4.0 * padded[3:-1]
            + padded[4:]
        ) / 16.0
        values = (
            (1.0 - frequency_smoothing) * values
            + frequency_smoothing * local_average
        )
    if startup_frames > 0:
        ramp = min(1.0, processed_frames / startup_frames)
        values = ramp * values + (1.0 - ramp)
    return values
