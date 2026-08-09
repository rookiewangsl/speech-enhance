"""Sohn-style statistical VAD with MCRA noise PSD tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from speech_frontend.noise.mcra import MCRA, MCRAConfig
from speech_frontend.vad.energy import VADResult
from speech_frontend.vad.state_machine import VADStateConfig, VADStateMachine


@dataclass(frozen=True)
class StatisticalVADConfig:
    """Causal likelihood-ratio VAD configuration."""

    sample_rate: int = 16_000
    frame_length: int = 320
    hop_length: int = 160
    fft_length: int = 512
    alpha_dd: float = 0.98
    score_threshold_on: float = 20.0
    score_threshold_off: float = 10.0
    score_probability_scale: float = 0.10
    onset_frames: int = 2
    minimum_speech_ms: int = 100
    hangover_ms: int = 200
    pre_roll_ms: int = 100
    minimum_frequency_hz: float = 80.0
    maximum_frequency_hz: float = 4_000.0
    maximum_snr: float = 1e4
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.frame_length <= 0 or self.hop_length <= 0:
            raise ValueError("sample rate, frame length, and hop length must be positive")
        if self.hop_length > self.frame_length or self.fft_length < self.frame_length:
            raise ValueError("invalid frame or FFT geometry")
        if not 0.0 <= self.alpha_dd < 1.0:
            raise ValueError("alpha_dd must be in [0, 1)")
        if self.score_threshold_on <= self.score_threshold_off:
            raise ValueError("score_threshold_on must exceed score_threshold_off")
        if self.score_probability_scale <= 0.0 or self.maximum_snr <= 0.0:
            raise ValueError("score scale and maximum SNR must be positive")
        if not 0.0 <= self.minimum_frequency_hz < self.maximum_frequency_hz:
            raise ValueError("invalid frequency range")


class StatisticalVAD:
    """Likelihood-ratio VAD using a decision-directed a priori SNR."""

    def __init__(
        self,
        config: StatisticalVADConfig | None = None,
        mcra_config: MCRAConfig | None = None,
    ) -> None:
        self.config = config or StatisticalVADConfig()
        self.mcra_config = mcra_config or MCRAConfig()
        frequencies = np.fft.rfftfreq(
            self.config.fft_length,
            d=1.0 / self.config.sample_rate,
        )
        self._frequency_mask = (
            (frequencies >= self.config.minimum_frequency_hz)
            & (frequencies <= self.config.maximum_frequency_hz)
        )
        if not np.any(self._frequency_mask):
            raise ValueError("frequency range contains no FFT bins")

    def detect(self, samples: NDArray[np.floating]) -> VADResult:
        """Return likelihood-derived VAD probabilities and stable segments."""

        signal = np.asarray(samples)
        if signal.ndim != 1:
            raise ValueError("samples must be one-dimensional")
        if not np.all(np.isfinite(signal)):
            raise ValueError("samples contain NaN or infinite values")
        spectra = self._spectra(signal)
        estimator = MCRA(self.mcra_config)
        previous_noise: NDArray[np.float64] | None = None
        previous_speech_power: NDArray[np.float64] | None = None
        scores: list[float] = []
        noise_trace: list[NDArray[np.float64]] = []

        for spectrum in spectra:
            noise_psd, _ = estimator.process_frame(spectrum)
            power = np.abs(spectrum) ** 2
            posterior = np.clip(
                power / np.maximum(noise_psd, self.config.epsilon),
                0.0,
                self.config.maximum_snr,
            )
            instantaneous_prior = np.maximum(posterior - 1.0, 0.0)
            if previous_noise is None:
                prior = instantaneous_prior
            else:
                prior = (
                    self.config.alpha_dd
                    * previous_speech_power
                    / np.maximum(previous_noise, self.config.epsilon)
                    + (1.0 - self.config.alpha_dd) * instantaneous_prior
                )
            prior = np.clip(prior, 0.0, self.config.maximum_snr)
            log_likelihood = (
                posterior * prior / (1.0 + prior) - np.log1p(prior)
            )
            scores.append(float(np.mean(log_likelihood[self._frequency_mask])))
            gain = prior / (1.0 + prior)
            previous_speech_power = gain**2 * power
            previous_noise = noise_psd
            noise_trace.append(noise_psd)

        score_array = np.asarray(scores)
        probability = 1.0 / (
            1.0
            + np.exp(
                -(score_array - self.config.score_threshold_on)
                / self.config.score_probability_scale
            )
        )
        hop_ms = 1_000.0 * self.config.hop_length / self.config.sample_rate
        labels = VADStateMachine(
            VADStateConfig(
                threshold_on=self.config.score_threshold_on,
                threshold_off=self.config.score_threshold_off,
                onset_frames=self.config.onset_frames,
                hangover_frames=round(self.config.hangover_ms / hop_ms),
                pre_roll_frames=round(self.config.pre_roll_ms / hop_ms),
                minimum_speech_frames=round(self.config.minimum_speech_ms / hop_ms),
            )
        ).apply(score_array)
        return VADResult(
            speech_probability=probability,
            frame_labels=labels,
            energy_db=score_array,
            noise_floor_db=10.0 * np.log10(
                np.mean(np.stack(noise_trace), axis=1) + self.config.epsilon
            ),
            snr_margin_db=score_array,
            segments=self._segments(labels, signal.size),
        )

    def _spectra(self, signal: NDArray[np.floating]) -> NDArray[np.complexfloating]:
        frame_count = (
            1
            if signal.size <= self.config.frame_length
            else int(
                np.ceil(
                    (signal.size - self.config.frame_length)
                    / self.config.hop_length
                )
            )
            + 1
        )
        padded_length = (
            (frame_count - 1) * self.config.hop_length
            + self.config.frame_length
        )
        padded = np.pad(
            signal.astype(np.float64, copy=False),
            (0, padded_length - signal.size),
        )
        shape = (frame_count, self.config.frame_length)
        strides = (
            padded.strides[0] * self.config.hop_length,
            padded.strides[0],
        )
        frames = np.lib.stride_tricks.as_strided(
            padded,
            shape=shape,
            strides=strides,
            writeable=False,
        )
        window = np.hanning(self.config.frame_length + 1)[:-1]
        return np.fft.rfft(frames * window, n=self.config.fft_length, axis=1)

    def _segments(
        self,
        labels: NDArray[np.bool_],
        sample_count: int,
    ) -> tuple[tuple[int, int], ...]:
        padded = np.pad(labels.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        return tuple(
            (
                int(start * self.config.hop_length),
                min(
                    sample_count,
                    int(
                        (stop - 1) * self.config.hop_length
                        + self.config.frame_length
                    ),
                ),
            )
            for start, stop in zip(starts, stops, strict=True)
        )
