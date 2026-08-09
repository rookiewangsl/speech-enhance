from __future__ import annotations

import numpy as np
import pytest

from speech_frontend.rnnoise.controller import (
    CorrectionAwareConfig,
    correction_aware_mix,
    fixed_residual_mix,
    vad_aware_mix,
)


def test_fixed_mix_endpoints() -> None:
    noisy = np.linspace(-0.5, 0.5, 1_000, dtype=np.float32)
    enhanced = 0.25 * noisy

    np.testing.assert_array_equal(
        fixed_residual_mix(noisy, enhanced, 0.0),
        noisy,
    )
    np.testing.assert_array_equal(
        fixed_residual_mix(noisy, enhanced, 1.0),
        enhanced,
    )


def test_vad_protection_moves_speech_frames_toward_input() -> None:
    noisy = np.ones(320, dtype=np.float32)
    enhanced = np.zeros(320, dtype=np.float32)
    vad = np.array([0.0, 1.0], dtype=np.float32)

    output = vad_aware_mix(
        noisy,
        enhanced,
        vad,
        speech_protection=0.5,
    )

    np.testing.assert_allclose(output[:160], 0.0)
    np.testing.assert_allclose(output[160:], 0.5)


def test_correction_aware_bypasses_identical_signals() -> None:
    rng = np.random.default_rng(5)
    noisy = rng.normal(scale=0.1, size=3_200).astype(np.float32)

    result = correction_aware_mix(
        noisy,
        noisy.copy(),
        config=CorrectionAwareConfig(initial_strength=0.0),
    )

    np.testing.assert_allclose(result.samples, noisy, atol=1e-7)
    assert np.all(result.frame_strength < 1e-6)


def test_correction_aware_uses_stronger_processing_for_large_change() -> None:
    noisy = np.full(3_200, 0.2, dtype=np.float32)
    enhanced = np.zeros_like(noisy)

    result = correction_aware_mix(
        noisy,
        enhanced,
        config=CorrectionAwareConfig(
            correction_threshold_db=-9.0,
            initial_strength=0.5,
        ),
    )

    assert result.frame_strength[-1] > 0.99
    assert np.mean(np.abs(result.samples[-320:])) < 0.005


def test_controller_output_is_finite_and_strength_is_bounded() -> None:
    rng = np.random.default_rng(9)
    noisy = rng.normal(scale=0.1, size=3_217).astype(np.float32)
    enhanced = (0.7 * noisy).astype(np.float32)
    vad = np.linspace(0.0, 1.0, 21, dtype=np.float32)

    result = correction_aware_mix(
        noisy,
        enhanced,
        vad_probabilities=vad,
        config=CorrectionAwareConfig(speech_protection=0.2),
    )

    assert result.samples.shape == noisy.shape
    assert np.all(np.isfinite(result.samples))
    assert np.all((result.frame_strength >= 0.0))
    assert np.all((result.frame_strength <= 1.0))


def test_controller_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="lengths"):
        correction_aware_mix(
            np.zeros(100, dtype=np.float32),
            np.zeros(99, dtype=np.float32),
        )
