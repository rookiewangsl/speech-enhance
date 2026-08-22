"""Causal VAD endpointing for dispatching live ASR segments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class EndpointConfig:
    """Frame-domain parameters for a live speech endpoint detector."""

    threshold_on: float = 0.60
    threshold_off: float = 0.35
    onset_frames: int = 2
    hangover_frames: int = 70
    pre_roll_frames: int = 10
    minimum_speech_frames: int = 20
    maximum_segment_frames: int = 1_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold_off < self.threshold_on <= 1.0:
            raise ValueError("VAD thresholds must satisfy 0 <= off < on <= 1")
        for name in (
            "onset_frames",
            "hangover_frames",
            "pre_roll_frames",
            "minimum_speech_frames",
            "maximum_segment_frames",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class EndpointSegment:
    """One causal ASR segment emitted after speech ends or reaches its limit."""

    samples: NDArray[np.float32]
    speech_frames: int
    total_frames: int
    reason: str
    start_frame: int
    end_frame: int


class StreamingEndpointDetector:
    """Turn one VAD score per audio frame into bounded ASR segments.

    The detector includes pre-roll and trailing hangover in each accepted
    segment.  It deliberately emits only completed segments: callers should
    call :meth:`flush` when microphone capture stops to dispatch an active
    final utterance.
    """

    def __init__(self, config: EndpointConfig | None = None) -> None:
        self.config = config or EndpointConfig()
        self._frame_samples: int | None = None
        self._frames_seen = 0
        self._pre_roll: deque[tuple[NDArray[np.float32], bool, int]] = deque(
            # Keep the preceding context plus the frames needed to confirm
            # onset; otherwise the onset frames would evict useful pre-roll.
            maxlen=self.config.pre_roll_frames + self.config.onset_frames
        )
        self._reset_active_state()

    def _reset_active_state(self) -> None:
        self._active = False
        self._onset_count = 0
        self._hangover_count = 0
        self._active_frames: list[NDArray[np.float32]] = []
        self._speech_frames = 0
        self._active_start_frame: int | None = None

    def _validate_frame(
        self,
        samples: FloatArray,
        probability: float,
    ) -> NDArray[np.float32]:
        frame = np.asarray(samples, dtype=np.float32)
        if frame.ndim != 1 or frame.size == 0:
            raise ValueError("endpoint frames must be non-empty one-dimensional audio")
        if not np.all(np.isfinite(frame)):
            raise ValueError("endpoint frame contains NaN or infinite values")
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("VAD probability must be finite and in [0, 1]")
        if self._frame_samples is None:
            self._frame_samples = int(frame.size)
        elif frame.size != self._frame_samples:
            raise ValueError("endpoint frame size changed during a stream")
        return frame.copy()

    def process_frame(
        self,
        samples: FloatArray,
        probability: float,
    ) -> EndpointSegment | None:
        """Consume one audio/VAD frame and return a completed segment if any."""

        frame = self._validate_frame(samples, probability)
        frame_index = self._frames_seen
        self._frames_seen += 1
        is_on = probability >= self.config.threshold_on
        is_speech = probability >= self.config.threshold_off

        if not self._active:
            self._pre_roll.append((frame, is_speech, frame_index))
            self._onset_count = self._onset_count + 1 if is_on else 0
            if self._onset_count < self.config.onset_frames:
                return None
            self._active = True
            self._onset_count = 0
            self._hangover_count = 0
            self._active_frames = [item[0] for item in self._pre_roll]
            self._speech_frames = sum(item[1] for item in self._pre_roll)
            self._active_start_frame = self._pre_roll[0][2]
            self._pre_roll.clear()
            return self._emit_if_maximum_reached()

        self._active_frames.append(frame)
        self._speech_frames += int(is_speech)
        self._hangover_count = (
            0 if is_speech else self._hangover_count + 1
        )
        if self._hangover_count >= self.config.hangover_frames:
            return self._finish("silence")
        return self._emit_if_maximum_reached()

    def _emit_if_maximum_reached(self) -> EndpointSegment | None:
        if len(self._active_frames) >= self.config.maximum_segment_frames:
            return self._finish("maximum_duration")
        return None

    def _finish(self, reason: str) -> EndpointSegment | None:
        frames = self._active_frames
        speech_frames = self._speech_frames
        start_frame = self._active_start_frame
        self._reset_active_state()
        if speech_frames < self.config.minimum_speech_frames:
            return None
        if start_frame is None:
            raise RuntimeError("endpoint segment has no start frame")
        return EndpointSegment(
            samples=np.concatenate(frames).astype(np.float32, copy=False),
            speech_frames=speech_frames,
            total_frames=len(frames),
            reason=reason,
            start_frame=start_frame,
            end_frame=start_frame + len(frames),
        )

    def flush(self) -> EndpointSegment | None:
        """Emit an active final utterance when its enclosing stream ends."""

        return self._finish("stream_end") if self._active else None
