from __future__ import annotations

import numpy as np
import pytest

from speech_frontend.stft import STFT, STFTConfig, STFTResult
from speech_frontend.streaming import StreamingWOLA


def run_in_chunks(
    processor: StreamingWOLA,
    signal: np.ndarray,
    chunk_sizes: list[int],
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    cursor = 0
    for size in chunk_sizes:
        if cursor >= signal.size:
            break
        outputs.append(processor.process_chunk(signal[cursor : cursor + size]))
        cursor += size
    if cursor < signal.size:
        outputs.append(processor.process_chunk(signal[cursor:]))
    outputs.append(processor.flush())
    return np.concatenate(outputs)


@pytest.mark.parametrize("length", [1, 127, 128, 511, 512, 8_003])
def test_streaming_identity_has_exact_length_and_values(length: int) -> None:
    rng = np.random.default_rng(101 + length)
    signal = rng.normal(size=length)
    stream = StreamingWOLA()

    output = run_in_chunks(stream, signal, [1, 31, 509, 7, 128, 2_000])

    assert output.shape == signal.shape
    np.testing.assert_allclose(output, signal, atol=1e-10, rtol=1e-10)


def test_streaming_matches_offline_for_fixed_spectral_gain() -> None:
    rng = np.random.default_rng(2026)
    signal = rng.normal(size=5_007)
    config = STFTConfig()
    transform = STFT(config)
    analysis = transform.analyze(signal)
    offline = transform.synthesize(
        STFTResult(
            spectrum=analysis.spectrum * 0.25,
            original_length=analysis.original_length,
            left_padding=analysis.left_padding,
        )
    )
    stream = StreamingWOLA(
        config,
        processor=lambda spectrum: spectrum * 0.25,
    )

    online = run_in_chunks(stream, signal, [17, 1_000, 3, 91, 2_048])

    np.testing.assert_allclose(online, offline, atol=1e-10, rtol=1e-10)


def test_flush_is_idempotent_and_push_after_flush_fails() -> None:
    stream = StreamingWOLA()
    stream.process_chunk(np.ones(1_000))
    stream.flush()

    assert stream.flush().size == 0
    with pytest.raises(RuntimeError, match="after flush"):
        stream.process_chunk(np.ones(1))


def test_reset_prevents_state_leak_between_files() -> None:
    rng = np.random.default_rng(7)
    first = rng.normal(size=2_000)
    second = rng.normal(size=3_000)
    stream = StreamingWOLA()
    run_in_chunks(stream, first, [333])
    stream.reset()

    output = run_in_chunks(stream, second, [29, 401, 5])

    np.testing.assert_allclose(output, second, atol=1e-10, rtol=1e-10)


def test_processor_must_preserve_spectrum_shape() -> None:
    stream = StreamingWOLA(processor=lambda spectrum: spectrum[:-1])

    with pytest.raises(ValueError, match="shape"):
        stream.process_chunk(np.ones(512))
