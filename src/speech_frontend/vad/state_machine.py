"""Hysteresis, pre-roll, and hangover for frame-level VAD scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VADStateConfig:
    """Frame-domain state parameters."""

    threshold_on: float
    threshold_off: float
    onset_frames: int = 2
    hangover_frames: int = 20
    pre_roll_frames: int = 10
    minimum_speech_frames: int = 10

    def __post_init__(self) -> None:
        if self.threshold_on <= self.threshold_off:
            raise ValueError("threshold_on must exceed threshold_off")
        for name in (
            "onset_frames",
            "hangover_frames",
            "pre_roll_frames",
            "minimum_speech_frames",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.onset_frames == 0:
            raise ValueError("onset_frames must be positive")


class VADStateMachine:
    """Convert causal continuous scores to stable frame labels."""

    def __init__(self, config: VADStateConfig) -> None:
        self.config = config

    def apply(self, scores: NDArray[np.floating]) -> NDArray[np.bool_]:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("scores must be one-dimensional")
        if not np.all(np.isfinite(values)):
            raise ValueError("scores contain NaN or infinite values")

        labels = np.zeros(values.size, dtype=np.bool_)
        active = False
        high_count = 0
        low_count = 0

        for index, score in enumerate(values):
            if not active:
                if score >= self.config.threshold_on:
                    high_count += 1
                else:
                    high_count = 0
                if high_count >= self.config.onset_frames:
                    active = True
                    start = max(
                        0,
                        index
                        - self.config.onset_frames
                        + 1
                        - self.config.pre_roll_frames,
                    )
                    labels[start : index + 1] = True
                    low_count = 0
            else:
                labels[index] = True
                if score < self.config.threshold_off:
                    low_count += 1
                else:
                    low_count = 0
                if low_count > self.config.hangover_frames:
                    labels[index] = False
                    active = False
                    high_count = 0
                    low_count = 0

        return self._remove_short_regions(labels)

    def _remove_short_regions(
        self,
        labels: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        minimum = self.config.minimum_speech_frames
        if minimum <= 1:
            return labels
        filtered = labels.copy()
        padded = np.pad(labels.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        for start, stop in zip(starts, stops, strict=True):
            if stop - start < minimum:
                filtered[start:stop] = False
        return filtered
