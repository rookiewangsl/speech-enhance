from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from speech_frontend.rnnoise import (
    RNNOISE_FRAME_SAMPLES,
    RNNoiseLibrary,
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


@pytest.fixture
def library() -> RNNoiseLibrary:
    return RNNoiseLibrary(LIBRARY_PATH)


def test_library_reports_expected_frame_size(
    library: RNNoiseLibrary,
) -> None:
    assert library.frame_size == RNNOISE_FRAME_SAMPLES


def test_silence_is_finite_and_vad_is_bounded(
    library: RNNoiseLibrary,
) -> None:
    with library.create_state() as state:
        output, vad_probability = state.process_frame(
            np.zeros(RNNOISE_FRAME_SAMPLES, dtype=np.float32)
        )

    assert output.shape == (RNNOISE_FRAME_SAMPLES,)
    assert np.all(np.isfinite(output))
    assert 0.0 <= vad_probability <= 1.0


@pytest.mark.parametrize("length", [0, 479, 481])
def test_process_frame_rejects_wrong_length(
    library: RNNoiseLibrary,
    length: int,
) -> None:
    with library.create_state() as state:
        with pytest.raises(ValueError, match="exactly 480"):
            state.process_frame(np.zeros(length, dtype=np.float32))


def test_process_frame_rejects_non_finite_and_out_of_range(
    library: RNNoiseLibrary,
) -> None:
    non_finite = np.zeros(RNNOISE_FRAME_SAMPLES, dtype=np.float32)
    non_finite[10] = np.nan
    out_of_range = np.zeros(RNNOISE_FRAME_SAMPLES, dtype=np.float32)
    out_of_range[10] = 1.01

    with library.create_state() as state:
        with pytest.raises(ValueError, match="NaN"):
            state.process_frame(non_finite)
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            state.process_frame(out_of_range)


def test_api_pcm_mode_matches_official_cli(
    library: RNNoiseLibrary,
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(20260724)
    samples = np.clip(
        0.20 * rng.normal(size=RNNOISE_FRAME_SAMPLES * 8),
        -1.0,
        1.0,
    ).astype(np.float32)
    input_pcm = np.rint(
        samples.astype(np.float64) * 32_767.0
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

    with library.create_state() as state:
        result = state.process_audio(samples, pcm_compatible=True)
    api_pcm = np.rint(result.samples * 32_767.0).astype("<i2")

    # The official demo intentionally discards the first processed 10 ms
    # frame as its fixed algorithmic-delay compensation.
    assert cli_pcm.size == api_pcm.size - RNNOISE_FRAME_SAMPLES
    np.testing.assert_array_equal(
        api_pcm[RNNOISE_FRAME_SAMPLES:],
        cli_pcm,
    )


def test_frame_by_frame_matches_process_audio(
    library: RNNoiseLibrary,
) -> None:
    rng = np.random.default_rng(11)
    samples = (0.1 * rng.normal(size=RNNOISE_FRAME_SAMPLES * 6)).astype(
        np.float32
    )

    with library.create_state() as state:
        whole = state.process_audio(samples)
    frames: list[np.ndarray] = []
    vad: list[float] = []
    with library.create_state() as state:
        for offset in range(0, samples.size, RNNOISE_FRAME_SAMPLES):
            output, probability = state.process_frame(
                samples[offset : offset + RNNOISE_FRAME_SAMPLES]
            )
            frames.append(output)
            vad.append(probability)

    np.testing.assert_array_equal(np.concatenate(frames), whole.samples)
    np.testing.assert_array_equal(
        np.asarray(vad, dtype=np.float32),
        whole.vad_probabilities,
    )


def test_reset_reproduces_fresh_state(library: RNNoiseLibrary) -> None:
    rng = np.random.default_rng(23)
    samples = (0.15 * rng.normal(size=RNNOISE_FRAME_SAMPLES * 5)).astype(
        np.float32
    )

    with library.create_state() as state:
        first = state.process_audio(samples)
        state.reset()
        second = state.process_audio(samples)

    np.testing.assert_array_equal(first.samples, second.samples)
    np.testing.assert_array_equal(
        first.vad_probabilities,
        second.vad_probabilities,
    )


def test_closed_state_rejects_processing(library: RNNoiseLibrary) -> None:
    state = library.create_state()
    state.close()
    state.close()

    with pytest.raises(RuntimeError, match="closed"):
        state.process_frame(
            np.zeros(RNNOISE_FRAME_SAMPLES, dtype=np.float32)
        )
