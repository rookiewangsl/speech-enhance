from __future__ import annotations

import numpy as np
import pytest

from speech_frontend.noise.mcra import MCRA, MCRAConfig
from speech_frontend.noise.imcra import IMCRA, IMCRAConfig


def test_mcra_noise_psd_and_probability_are_finite_and_bounded() -> None:
    rng = np.random.default_rng(3)
    spectra = rng.normal(size=(30, 17)) + 1j * rng.normal(size=(30, 17))

    noise, probability = MCRA().process_spectra(spectra)

    assert noise.shape == probability.shape == spectra.shape
    assert np.all(np.isfinite(noise))
    assert np.all(noise >= 0.0)
    assert np.all((0.0 <= probability) & (probability <= 1.0))


def test_mcra_speech_probability_slows_noise_updates() -> None:
    estimator = MCRA(
        MCRAConfig(
            alpha_d=0.95,
            alpha_p=0.2,
            alpha_s=0.0,
            minimum_window_frames=20,
            ratio_threshold=2.0,
        )
    )
    for _ in range(10):
        baseline_noise, _ = estimator.process_frame(np.ones(5))
    after_peak, probability = estimator.process_frame(np.full(5, 10.0))

    assert np.all(probability > 0.5)
    assert np.all(after_peak < baseline_noise + 15.0)


def test_mcra_rejects_changed_bin_count_without_reset() -> None:
    estimator = MCRA()
    estimator.process_frame(np.ones(5))

    with pytest.raises(ValueError, match="frequency-bin"):
        estimator.process_frame(np.ones(6))


def test_imcra_noise_psd_and_probability_are_finite_and_bounded() -> None:
    rng = np.random.default_rng(31)
    estimator = IMCRA(IMCRAConfig(subwindow_frames=5, history_subwindows=3))

    results = [
        estimator.process_frame(
            rng.normal(size=17) + 1j * rng.normal(size=17),
        )
        for _ in range(30)
    ]
    noise = np.stack([result[0] for result in results])
    probability = np.stack([result[1] for result in results])

    assert noise.shape == probability.shape == (30, 17)
    assert np.all(np.isfinite(noise))
    assert np.all(noise >= 0.0)
    assert np.all((0.0 <= probability) & (probability <= 1.0))


def test_imcra_reset_restores_first_frame_behavior() -> None:
    estimator = IMCRA()
    spectrum = np.full(9, 2.0 + 1.0j)

    first_noise, first_probability = estimator.process_frame(spectrum)
    estimator.process_frame(spectrum)
    estimator.reset()
    reset_noise, reset_probability = estimator.process_frame(spectrum)

    np.testing.assert_allclose(reset_noise, first_noise)
    np.testing.assert_allclose(reset_probability, first_probability)
