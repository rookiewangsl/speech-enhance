from __future__ import annotations

import numpy as np

from speech_frontend.rnnoise import (
    RESAMPLER_GROUP_DELAY_48K,
    StreamingDownsampler3,
    StreamingUpsampler3,
)


def process_chunks(processor: object, samples: np.ndarray) -> np.ndarray:
    outputs: list[np.ndarray] = []
    cursor = 0
    for size in [1, 127, 3, 900, 2, 2_001]:
        if cursor >= samples.size:
            break
        outputs.append(
            processor.process_chunk(samples[cursor : cursor + size])
        )
        cursor += size
    if cursor < samples.size:
        outputs.append(processor.process_chunk(samples[cursor:]))
    return np.concatenate(outputs)


def test_upsampler_is_chunk_invariant() -> None:
    rng = np.random.default_rng(77)
    samples = rng.normal(scale=0.1, size=4_003).astype(np.float32)

    reference = StreamingUpsampler3().process_chunk(samples)
    chunked = process_chunks(StreamingUpsampler3(), samples)

    assert reference.size == samples.size * 3
    np.testing.assert_array_equal(chunked, reference)


def test_downsampler_is_chunk_invariant() -> None:
    rng = np.random.default_rng(79)
    samples = rng.normal(scale=0.1, size=12_009).astype(np.float32)

    reference = StreamingDownsampler3().process_chunk(samples)
    chunked = process_chunks(StreamingDownsampler3(), samples)

    assert reference.size == samples.size // 3
    np.testing.assert_array_equal(chunked, reference)


def test_round_trip_has_expected_causal_delay_and_length() -> None:
    impulse = np.zeros(2_000, dtype=np.float32)
    impulse[200] = 1.0
    upsampled = StreamingUpsampler3().process_chunk(impulse)
    reconstructed = StreamingDownsampler3().process_chunk(upsampled)
    expected_delay = 2 * RESAMPLER_GROUP_DELAY_48K // 3

    assert reconstructed.shape == impulse.shape
    assert np.argmax(np.abs(reconstructed)) == 200 + expected_delay
    assert expected_delay == 42


def test_round_trip_preserves_passband_tone_after_delay() -> None:
    time = np.arange(16_000, dtype=np.float64) / 16_000
    samples = (
        0.3 * np.sin(2 * np.pi * 440 * time)
        + 0.2 * np.sin(2 * np.pi * 3_000 * time)
    ).astype(np.float32)
    upsampled = StreamingUpsampler3().process_chunk(samples)
    reconstructed = StreamingDownsampler3().process_chunk(upsampled)
    delay = 2 * RESAMPLER_GROUP_DELAY_48K // 3
    valid_reference = samples[200:-delay]
    valid_reconstructed = reconstructed[200 + delay :]
    error = valid_reconstructed - valid_reference

    assert np.sqrt(np.mean(error**2)) < 2e-4


def test_reset_reproduces_upsampler_output() -> None:
    rng = np.random.default_rng(7)
    samples = rng.normal(scale=0.1, size=1_003).astype(np.float32)
    upsampler = StreamingUpsampler3()
    first = upsampler.process_chunk(samples)
    upsampler.reset()
    second = upsampler.process_chunk(samples)

    np.testing.assert_array_equal(first, second)
