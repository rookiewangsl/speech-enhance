from __future__ import annotations

import numpy as np
import pytest

from speech_frontend.vad.endpoint import EndpointConfig, StreamingEndpointDetector


def frame(value: float) -> np.ndarray:
    return np.full(160, value, dtype=np.float32)


def test_endpoint_includes_pre_roll_and_hangover() -> None:
    detector = StreamingEndpointDetector(
        EndpointConfig(
            threshold_on=0.6,
            threshold_off=0.4,
            onset_frames=2,
            hangover_frames=2,
            pre_roll_frames=2,
            minimum_speech_frames=3,
            maximum_segment_frames=20,
        )
    )

    emitted = [
        detector.process_frame(samples, probability)
        for samples, probability in (
            (frame(1), 0.1),
            (frame(2), 0.8),
            (frame(3), 0.9),
            (frame(4), 0.7),
            (frame(5), 0.1),
            (frame(6), 0.1),
        )
    ]

    segment = emitted[-1]
    assert segment is not None
    assert segment.reason == "silence"
    assert segment.speech_frames == 3
    assert segment.total_frames == 6
    assert (segment.start_frame, segment.end_frame) == (0, 6)
    np.testing.assert_array_equal(segment.samples, np.concatenate([frame(i) for i in range(1, 7)]))


def test_endpoint_discards_short_speech_and_can_flush_active_speech() -> None:
    detector = StreamingEndpointDetector(
        EndpointConfig(
            onset_frames=1,
            hangover_frames=1,
            pre_roll_frames=1,
            minimum_speech_frames=2,
            maximum_segment_frames=20,
        )
    )

    assert detector.process_frame(frame(1), 0.9) is None
    assert detector.process_frame(frame(2), 0.1) is None
    assert detector.process_frame(frame(3), 0.9) is None
    segment = detector.process_frame(frame(4), 0.9)
    assert segment is None
    final = detector.flush()

    assert final is not None
    assert final.reason == "stream_end"
    assert final.speech_frames == 2
    assert (final.start_frame, final.end_frame) == (2, 4)
    np.testing.assert_array_equal(final.samples, np.concatenate((frame(3), frame(4))))


def test_endpoint_rejects_inconsistent_frame_sizes_and_invalid_config() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        EndpointConfig(threshold_on=0.3, threshold_off=0.3)

    detector = StreamingEndpointDetector()
    detector.process_frame(np.zeros(160, dtype=np.float32), 0.0)
    with pytest.raises(ValueError, match="size changed"):
        detector.process_frame(np.zeros(80, dtype=np.float32), 0.0)
