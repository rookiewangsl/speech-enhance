from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from speech_frontend.rnnoise import (
    RNNOISE_FRAME_SAMPLES,
    RNNoiseLibrary,
    StreamingRNNoise48k,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = (
    PROJECT_ROOT / "build" / "rnnoise" / "lib" / "librnnoise.dylib"
)
CLI_PATH = PROJECT_ROOT / "build" / "rnnoise" / "bin" / "rnnoise_demo"
RNNOISE_BUILT = LIBRARY_PATH.is_file() and CLI_PATH.is_file()

pytestmark = pytest.mark.skipif(
    not RNNOISE_BUILT,
    reason="run scripts/setup_rnnoise.sh before RNNoise integration tests",
)


def run_chunks(
    stream: StreamingRNNoise48k,
    samples: np.ndarray,
    chunk_sizes: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    output: list[np.ndarray] = []
    vad: list[np.ndarray] = []
    cursor = 0
    for chunk_size in chunk_sizes:
        if cursor >= samples.size:
            break
        result = stream.process_chunk(
            samples[cursor : cursor + chunk_size]
        )
        output.append(result.samples)
        vad.append(result.vad_probabilities)
        cursor += chunk_size
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


@pytest.mark.parametrize(
    "length",
    [1, 479, 480, 481, 1_003, RNNOISE_FRAME_SAMPLES * 8],
)
def test_streaming_output_has_exact_input_length(
    library: RNNoiseLibrary,
    length: int,
) -> None:
    rng = np.random.default_rng(100 + length)
    samples = (0.1 * rng.normal(size=length)).astype(np.float32)
    stream = StreamingRNNoise48k(library)

    output, _ = run_chunks(stream, samples, [1, 127, 480, 19])

    assert output.shape == samples.shape
    assert stream.total_input_samples == length
    assert stream.total_output_samples == length
    assert np.all(np.isfinite(output))


def test_arbitrary_chunks_match_one_chunk(
    library: RNNoiseLibrary,
) -> None:
    rng = np.random.default_rng(888)
    samples = (0.12 * rng.normal(size=8_003)).astype(np.float32)

    one_chunk, one_vad = run_chunks(
        StreamingRNNoise48k(library),
        samples,
        [samples.size],
    )
    arbitrary, arbitrary_vad = run_chunks(
        StreamingRNNoise48k(library),
        samples,
        [1, 127, 480, 3, 1_001, 19, 2_000],
    )

    np.testing.assert_array_equal(arbitrary, one_chunk)
    np.testing.assert_array_equal(arbitrary_vad, one_vad)


def test_pcm_stream_matches_official_cli_and_zero_tail(
    library: RNNoiseLibrary,
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(404)
    samples = (0.08 * rng.normal(size=4_123)).astype(np.float32)
    padded_size = (
        (samples.size + RNNOISE_FRAME_SAMPLES - 1)
        // RNNOISE_FRAME_SAMPLES
        * RNNOISE_FRAME_SAMPLES
    )
    padded = np.pad(samples, (0, padded_size - samples.size))
    input_pcm = np.rint(
        padded.astype(np.float64) * 32_767.0
    ).astype("<i2")
    input_path = tmp_path / "input.raw"
    output_path = tmp_path / "output.raw"
    input_pcm.tofile(input_path)
    subprocess.run(
        [str(CLI_PATH), str(input_path), str(output_path)],
        check=True,
        capture_output=True,
    )
    cli_pcm = np.fromfile(output_path, dtype="<i2")
    expected_pcm = np.concatenate(
        (
            cli_pcm,
            np.zeros(samples.size - cli_pcm.size, dtype=np.int16),
        )
    )

    stream = StreamingRNNoise48k(library, pcm_compatible=True)
    output, _ = run_chunks(stream, samples, [31, 700, 2, 1_001])
    actual_pcm = np.rint(output * 32_767.0).astype("<i2")

    np.testing.assert_array_equal(actual_pcm, expected_pcm)


def test_first_frame_is_held_for_ten_ms_delay(
    library: RNNoiseLibrary,
) -> None:
    stream = StreamingRNNoise48k(library)
    frame = np.zeros(RNNOISE_FRAME_SAMPLES, dtype=np.float32)

    first = stream.process_chunk(frame)
    second = stream.process_chunk(frame)

    assert first.samples.size == 0
    assert second.samples.size == RNNOISE_FRAME_SAMPLES
    assert stream.algorithmic_delay_samples == RNNOISE_FRAME_SAMPLES
    assert stream.alignment_delay_samples == RNNOISE_FRAME_SAMPLES


def test_flush_is_idempotent_and_push_after_flush_fails(
    library: RNNoiseLibrary,
) -> None:
    stream = StreamingRNNoise48k(library)
    stream.process_chunk(np.zeros(700, dtype=np.float32))
    first = stream.flush()
    second = stream.flush()

    assert first.samples.size > 0
    assert second.samples.size == 0
    with pytest.raises(RuntimeError, match="after flush"):
        stream.process_chunk(np.zeros(1, dtype=np.float32))


def test_reset_reproduces_fresh_stream(
    library: RNNoiseLibrary,
) -> None:
    rng = np.random.default_rng(91)
    samples = (0.1 * rng.normal(size=3_003)).astype(np.float32)
    stream = StreamingRNNoise48k(library)
    first, first_vad = run_chunks(stream, samples, [17, 901])
    stream.reset()
    second, second_vad = run_chunks(stream, samples, [480, 11, 2_000])

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_vad, second_vad)
