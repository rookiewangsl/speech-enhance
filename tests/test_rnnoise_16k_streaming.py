from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from speech_frontend.rnnoise import RNNoiseLibrary, StreamingRNNoise16k

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = (
    PROJECT_ROOT / "build" / "rnnoise" / "lib" / "librnnoise.dylib"
)
RNNOISE_BUILT = LIBRARY_PATH.is_file()

pytestmark = pytest.mark.skipif(
    not RNNOISE_BUILT,
    reason="run scripts/setup_rnnoise.sh before RNNoise integration tests",
)


def run_chunks(
    stream: StreamingRNNoise16k,
    samples: np.ndarray,
    chunk_sizes: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    output: list[np.ndarray] = []
    vad: list[np.ndarray] = []
    cursor = 0
    for size in chunk_sizes:
        if cursor >= samples.size:
            break
        result = stream.process_chunk(samples[cursor : cursor + size])
        output.append(result.samples)
        vad.append(result.vad_probabilities)
        cursor += size
    if cursor < samples.size:
        result = stream.process_chunk(samples[cursor:])
        output.append(result.samples)
        vad.append(result.vad_probabilities)
    result = stream.flush()
    output.append(result.samples)
    vad.append(result.vad_probabilities)
    return np.concatenate(output), np.concatenate(vad)


@pytest.fixture
def library() -> RNNoiseLibrary:
    return RNNoiseLibrary(LIBRARY_PATH)


@pytest.mark.parametrize("length", [1, 159, 160, 161, 1_003, 8_001])
def test_16k_stream_has_exact_length(
    library: RNNoiseLibrary,
    length: int,
) -> None:
    rng = np.random.default_rng(800 + length)
    samples = rng.normal(scale=0.05, size=length).astype(np.float32)
    stream = StreamingRNNoise16k(library)

    output, _ = run_chunks(stream, samples, [1, 17, 160, 701])

    assert output.shape == samples.shape
    assert stream.total_input_samples == length
    assert stream.total_output_samples == length
    assert np.all(np.isfinite(output))


def test_16k_arbitrary_chunks_match_one_chunk(
    library: RNNoiseLibrary,
) -> None:
    rng = np.random.default_rng(108)
    samples = rng.normal(scale=0.08, size=9_017).astype(np.float32)

    one, one_vad = run_chunks(
        StreamingRNNoise16k(library),
        samples,
        [samples.size],
    )
    chunked, chunked_vad = run_chunks(
        StreamingRNNoise16k(library),
        samples,
        [1, 13, 159, 2, 700, 2_001],
    )

    np.testing.assert_array_equal(chunked, one)
    np.testing.assert_array_equal(chunked_vad, one_vad)


def test_16k_reset_reproduces_fresh_stream(
    library: RNNoiseLibrary,
) -> None:
    rng = np.random.default_rng(110)
    samples = rng.normal(scale=0.08, size=3_007).astype(np.float32)
    stream = StreamingRNNoise16k(library)
    first, first_vad = run_chunks(stream, samples, [17, 501])
    stream.reset()
    second, second_vad = run_chunks(stream, samples, [160, 9, 1_200])

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_vad, second_vad)
    assert stream.algorithmic_delay_samples == 202
    assert stream.alignment_delay_samples == 202
