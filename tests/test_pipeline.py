from __future__ import annotations

import numpy as np
import pytest

from speech_frontend.pipeline import ClassicalEnhancer


def test_identity_pipeline_is_exact() -> None:
    rng = np.random.default_rng(21)
    signal = rng.normal(size=3_003)

    result = ClassicalEnhancer().enhance(signal, method="identity")

    np.testing.assert_allclose(result.samples, signal, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result.diagnostics.gain, 1.0)


@pytest.mark.parametrize(
    "method",
    [
        "spectral_subtraction",
        "mcra_instantaneous_wiener",
        "mcra_dd_wiener",
        "mcra_om_lsa",
        "imcra_om_lsa",
        "dual_uncertainty_wiener",
    ],
)
def test_enhancement_baselines_preserve_length_and_finite_values(
    method: str,
) -> None:
    rng = np.random.default_rng(22)
    signal = rng.normal(size=3_003)

    result = ClassicalEnhancer().enhance(signal, method=method)

    assert result.samples.shape == signal.shape
    assert np.all(np.isfinite(result.samples))
    assert np.all(np.isfinite(result.diagnostics.noise_psd))
    assert np.all((0.05 <= result.diagnostics.gain) & (result.diagnostics.gain <= 1.0))
    if method in {"mcra_om_lsa", "imcra_om_lsa", "dual_uncertainty_wiener"}:
        assert result.diagnostics.gain_speech_probability is not None
        assert np.all(
            (0.0 <= result.diagnostics.gain_speech_probability)
            & (result.diagnostics.gain_speech_probability <= 1.0)
        )
