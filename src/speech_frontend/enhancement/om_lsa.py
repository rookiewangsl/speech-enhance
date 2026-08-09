"""OM-LSA spectral gain with explicit speech-presence uncertainty.

This implements the gain-side component of Cohen and Berdugo (2001).  MCRA
remains responsible for causal noise-PSD tracking; this class replaces the
Wiener gain when isolated suppression notches cause musical noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import exp1

ComplexArray = NDArray[np.complexfloating]
FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class OMLSAConfig:
    """Numerically safe settings for one causal OM-LSA processor."""

    alpha_dd: float = 0.92
    gain_floor: float = 0.10
    speech_absence_prior_min: float = 0.05
    speech_absence_prior_max: float = 0.95
    maximum_snr: float = 1e4
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha_dd < 1.0:
            raise ValueError("alpha_dd must be in [0, 1)")
        if not 0.0 < self.gain_floor <= 1.0:
            raise ValueError("gain_floor must be in (0, 1]")
        if not (
            0.0
            < self.speech_absence_prior_min
            < self.speech_absence_prior_max
            < 1.0
        ):
            raise ValueError("speech-absence prior bounds must lie in (0, 1)")
        if self.maximum_snr <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("maximum_snr and epsilon must be positive")


class OMLSA:
    """Stateful causal OM-LSA gain estimator.

    ``local_speech_presence_probability`` comes from MCRA and is used only as
    a local prior for the speech-absence hypothesis.  It is deliberately not a
    hard VAD gate.  The result is the geometric mixture of the LSA gain under
    speech presence and the configured gain floor under speech absence.
    """

    def __init__(self, config: OMLSAConfig | None = None) -> None:
        self.config = config or OMLSAConfig()
        self.reset()

    def reset(self) -> None:
        """Clear utterance-local DD state."""

        self._previous_gain: NDArray[np.float64] | None = None
        self._previous_posterior_snr: NDArray[np.float64] | None = None

    def process_frame(
        self,
        noisy_spectrum: ComplexArray,
        noise_psd: FloatArray,
        *,
        local_speech_presence_probability: FloatArray,
    ) -> tuple[ComplexArray, NDArray[np.float64], NDArray[np.float64]]:
        """Return enhanced spectrum, OM-LSA gain, and posterior speech probability."""

        spectrum = np.asarray(noisy_spectrum)
        noise = np.asarray(noise_psd, dtype=np.float64)
        local_presence = np.asarray(
            local_speech_presence_probability,
            dtype=np.float64,
        )
        if spectrum.ndim != 1 or noise.shape != spectrum.shape:
            raise ValueError("one spectrum and matching noise PSD are required")
        if local_presence.shape != spectrum.shape:
            raise ValueError("speech-presence probability must match spectrum")
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("spectrum contains NaN or infinite values")
        if not np.all(np.isfinite(noise)) or np.any(noise < 0.0):
            raise ValueError("noise_psd must be finite and non-negative")
        if not np.all(np.isfinite(local_presence)):
            raise ValueError("speech-presence probability must be finite")

        posterior = np.clip(
            np.abs(spectrum) ** 2 / np.maximum(noise, self.config.epsilon),
            0.0,
            self.config.maximum_snr,
        )
        instantaneous_prior = np.maximum(posterior - 1.0, 0.0)
        if self._previous_gain is None:
            prior = instantaneous_prior
        else:
            if self._previous_gain.shape != spectrum.shape:
                raise ValueError("frequency-bin count changed without reset")
            prior = (
                self.config.alpha_dd
                * self._previous_gain**2
                * self._previous_posterior_snr
                + (1.0 - self.config.alpha_dd) * instantaneous_prior
            )
        prior = np.clip(prior, 0.0, self.config.maximum_snr)

        # q(k,l) is the a-priori speech-absence probability.  MCRA estimates
        # local speech presence, so its complement provides a causal prior.
        absence_prior = np.clip(
            1.0 - np.clip(local_presence, 0.0, 1.0),
            self.config.speech_absence_prior_min,
            self.config.speech_absence_prior_max,
        )
        v = posterior * prior / (1.0 + prior)
        log_lsa_gain = (
            np.log(np.maximum(prior, self.config.epsilon))
            - np.log1p(prior)
            + 0.5 * exp1(np.maximum(v, self.config.epsilon))
        )
        log_lsa_gain = np.clip(log_lsa_gain, -60.0, 0.0)

        log_absence_to_presence = (
            np.log(absence_prior)
            - np.log1p(-absence_prior)
            + np.log1p(prior)
            - v
        )
        speech_probability = 1.0 / (
            1.0 + np.exp(np.clip(log_absence_to_presence, -60.0, 60.0))
        )
        log_gain = (
            speech_probability * log_lsa_gain
            + (1.0 - speech_probability) * np.log(self.config.gain_floor)
        )
        gain = np.clip(
            np.exp(np.clip(log_gain, -60.0, 0.0)),
            self.config.gain_floor,
            1.0,
        )
        self._previous_gain = gain
        self._previous_posterior_snr = posterior
        return gain * spectrum, gain.copy(), speech_probability.copy()
