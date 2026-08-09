from __future__ import annotations

import numpy as np
import pytest

from speech_frontend.rnnoise.continuity import (
    ContinuityConfig,
    stabilize_output_continuity,
)


def test_constant_voiced_signal_is_unchanged() -> None:
    samples = np.full(640, 0.1, dtype=np.float32)
    vad = np.ones(4, dtype=np.float32)

    result = stabilize_output_continuity(samples, vad)

    np.testing.assert_allclose(result.samples, samples, atol=1e-7)
    np.testing.assert_array_equal(result.frame_gain_db, 0.0)


def test_sudden_voiced_drop_receives_bounded_boost() -> None:
    samples = np.concatenate(
        (
            np.full(160, 0.1, dtype=np.float32),
            np.full(160, 0.01, dtype=np.float32),
        )
    )
    vad = np.ones(2, dtype=np.float32)
    settings = ContinuityConfig(
        allowed_drop_db=3.0,
        envelope_release_db_per_frame=0.0,
        max_boost_db=6.0,
    )

    result = stabilize_output_continuity(samples, vad, config=settings)

    assert result.frame_gain_db[0] == pytest.approx(0.0)
    assert result.frame_gain_db[1] == pytest.approx(6.0)
    assert np.mean(np.abs(result.samples[160:])) > 0.01


def test_non_speech_drop_is_not_boosted() -> None:
    samples = np.concatenate(
        (
            np.full(160, 0.1, dtype=np.float32),
            np.full(160, 0.01, dtype=np.float32),
        )
    )
    vad = np.array([1.0, 0.0], dtype=np.float32)

    result = stabilize_output_continuity(samples, vad)

    assert result.frame_gain_db[1] == pytest.approx(0.0)
    np.testing.assert_allclose(result.samples, samples, atol=1e-7)


def test_result_preserves_length_and_bounds_gain() -> None:
    rng = np.random.default_rng(8)
    samples = rng.normal(scale=0.05, size=1_777).astype(np.float32)
    vad = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    settings = ContinuityConfig(max_boost_db=4.0)

    result = stabilize_output_continuity(samples, vad, config=settings)

    assert result.samples.shape == samples.shape
    assert np.all(np.isfinite(result.samples))
    assert np.all(result.frame_gain_db >= 0.0)
    assert np.all(result.frame_gain_db <= 4.0)


def test_invalid_vad_is_rejected() -> None:
    samples = np.zeros(160, dtype=np.float32)

    with pytest.raises(ValueError, match="VAD"):
        stabilize_output_continuity(samples, np.array([], dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        stabilize_output_continuity(
            samples,
            np.array([1.1], dtype=np.float32),
        )
