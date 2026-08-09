"""Adaptive log-energy voice activity detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from speech_frontend.vad.state_machine import (
    VADStateConfig,
    VADStateMachine,
)

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class EnergyVADConfig:
    """Parameters for the causal adaptive-energy baseline."""

    sample_rate: int = 16_000
    frame_length: int = 320
    hop_length: int = 160
    noise_history_seconds: float = 2.0
    noise_percentile: float = 20.0
    initial_noise_floor_db: float = -55.0
    noise_floor_rise_smoothing: float = 0.98
    noise_floor_fall_smoothing: float = 0.8
    threshold_on_db: float = 12.0
    threshold_off_db: float = 9.0
    onset_frames: int = 2
    minimum_speech_ms: int = 100
    hangover_ms: int = 200
    pre_roll_ms: int = 100
    probability_midpoint_db: float = 6.0
    probability_scale_db: float = 2.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_length <= 0 or self.hop_length <= 0:
            raise ValueError("frame and hop lengths must be positive")
        if self.hop_length > self.frame_length:
            raise ValueError("hop_length cannot exceed frame_length")
        if self.noise_history_seconds <= 0:
            raise ValueError("noise_history_seconds must be positive")
        if not 0.0 <= self.noise_percentile <= 100.0:
            raise ValueError("noise_percentile must be in [0, 100]")
        for value in (
            self.noise_floor_rise_smoothing,
            self.noise_floor_fall_smoothing,
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError("noise smoothing must be in [0, 1)")
        if self.probability_scale_db <= 0:
            raise ValueError("probability_scale_db must be positive")


@dataclass(frozen=True)
class VADResult:
    """Frame-level diagnostics and sample-domain speech regions."""

    speech_probability: NDArray[np.float64]
    frame_labels: NDArray[np.bool_]
    energy_db: NDArray[np.float64]
    noise_floor_db: NDArray[np.float64]
    snr_margin_db: NDArray[np.float64]
    segments: tuple[tuple[int, int], ...]


class EnergyVAD:
    """Adaptive log-energy VAD with no fixed leading-noise assumption."""

    def __init__(self, config: EnergyVADConfig | None = None) -> None:
        self.config = config or EnergyVADConfig()

    def detect(self, samples: FloatArray) -> VADResult:
        """Run VAD and return probabilities, labels, and sample intervals."""

        signal = np.asarray(samples)
        if signal.ndim != 1:
            raise ValueError("samples must be one-dimensional")
        if not np.all(np.isfinite(signal)):
            raise ValueError("samples contain NaN or infinite values")

        frames = self._frame(signal)
        energy = 10.0 * np.log10(
            np.mean(frames**2, axis=1) + np.finfo(np.float64).tiny
        )
        energy = np.maximum(energy, -120.0)
        noise_floor = self._track_noise_floor(energy)
        margin = energy - noise_floor
        probability = 1.0 / (
            1.0
            + np.exp(
                -(
                    margin - self.config.probability_midpoint_db
                )
                / self.config.probability_scale_db
            )
        )

        hop_ms = 1000.0 * self.config.hop_length / self.config.sample_rate
        state = VADStateMachine(
            VADStateConfig(
                threshold_on=self.config.threshold_on_db,
                threshold_off=self.config.threshold_off_db,
                onset_frames=self.config.onset_frames,
                hangover_frames=round(self.config.hangover_ms / hop_ms),
                pre_roll_frames=round(self.config.pre_roll_ms / hop_ms),
                minimum_speech_frames=round(
                    self.config.minimum_speech_ms / hop_ms
                ),
            )
        )
        labels = state.apply(margin)
        segments = self._segments(labels, signal.size)
        return VADResult(
            speech_probability=probability,
            frame_labels=labels,
            energy_db=energy,
            noise_floor_db=noise_floor,
            snr_margin_db=margin,
            segments=segments,
        )

    def _frame(self, signal: FloatArray) -> NDArray[np.float64]:
        frame = self.config.frame_length
        hop = self.config.hop_length
        frame_count = (
            1
            if signal.size <= frame
            else int(np.ceil((signal.size - frame) / hop)) + 1
        )
        padded_length = (frame_count - 1) * hop + frame
        padded = np.pad(
            signal.astype(np.float64, copy=False),
            (0, padded_length - signal.size),
        )
        shape = (frame_count, frame)
        strides = (padded.strides[0] * hop, padded.strides[0])
        return np.lib.stride_tricks.as_strided(
            padded,
            shape=shape,
            strides=strides,
            writeable=False,
        )

    def _track_noise_floor(
        self,
        energy_db: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        history_length = max(
            1,
            round(
                self.config.noise_history_seconds
                * self.config.sample_rate
                / self.config.hop_length
            ),
        )
        floor = np.empty_like(energy_db)
        current = self.config.initial_noise_floor_db
        for index, energy in enumerate(energy_db):
            start = max(0, index - history_length + 1)
            candidate = float(
                np.percentile(
                    energy_db[start : index + 1],
                    self.config.noise_percentile,
                )
            )
            alpha = (
                self.config.noise_floor_rise_smoothing
                if candidate > current
                else self.config.noise_floor_fall_smoothing
            )
            current = alpha * current + (1.0 - alpha) * candidate
            floor[index] = current
        return floor

    def _segments(
        self,
        labels: NDArray[np.bool_],
        sample_count: int,
    ) -> tuple[tuple[int, int], ...]:
        padded = np.pad(labels.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        segments: list[tuple[int, int]] = []
        for start, stop in zip(starts, stops, strict=True):
            start_sample = int(start * self.config.hop_length)
            stop_sample = min(
                sample_count,
                int(
                    (stop - 1) * self.config.hop_length
                    + self.config.frame_length
                ),
            )
            segments.append((start_sample, stop_sample))
        return tuple(segments)
