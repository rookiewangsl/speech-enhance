from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from robust_asr.dereverb.wpe import (
    WPEConfig,
    analyze_multichannel,
    build_delayed_history,
    offline_wpe_spectrum,
    offline_wpe_waveform,
    synthesize_multichannel,
)


def test_delayed_history_uses_channel_tap_blocks() -> None:
    spectrum = np.arange(1, 13).reshape(1, 2, 6).astype(np.complex128)

    history = build_delayed_history(spectrum, taps=2, delay=1)

    assert history.shape == (1, 4, 6)
    np.testing.assert_array_equal(history[0, 0:2, 1:], spectrum[0, :, :-1])
    np.testing.assert_array_equal(history[0, 2:4, 2:], spectrum[0, :, :-2])


def test_formal_hann_analysis_synthesis_round_trip() -> None:
    rng = np.random.default_rng(9)
    signals = rng.normal(size=(4, 4_123))
    config = WPEConfig(
        n_fft=256,
        win_length=256,
        hop_length=64,
        delay=2,
        taps=5,
    )

    spectrum = analyze_multichannel(signals, config)
    reconstructed = synthesize_multichannel(
        spectrum, config, output_length=signals.shape[1]
    )

    np.testing.assert_allclose(reconstructed, signals, atol=1e-10, rtol=1e-10)


def test_numpy_reference_wpe_preserves_waveform_shape_and_finiteness() -> None:
    rng = np.random.default_rng(2026)
    source = rng.normal(scale=0.05, size=4_096)
    signals = np.stack(
        [
            np.convolve(source, np.array([1.0, 0.0, 0.3 + 0.03 * channel]))[
                : source.size
            ]
            for channel in range(2)
        ]
    )
    config = WPEConfig(
        n_fft=128,
        win_length=128,
        hop_length=32,
        delay=1,
        taps=2,
        iterations=1,
    )

    output = offline_wpe_waveform(
        signals, config, backend="numpy_reference"
    )

    assert output.shape == signals.shape
    assert np.all(np.isfinite(output))


def test_reference_wpe_rejects_too_few_frames() -> None:
    spectrum = np.zeros((10, 2, 4), dtype=np.complex128)
    config = WPEConfig(delay=2, taps=3)

    with pytest.raises(ValueError, match="requires more than"):
        offline_wpe_spectrum(spectrum, config, backend="numpy_reference")


def test_formal_backend_reports_missing_optional_dependency() -> None:
    if importlib.util.find_spec("nara_wpe") is not None:
        pytest.skip("nara_wpe is installed in this environment")
    spectrum = np.zeros((20, 2, 20), dtype=np.complex128)

    with pytest.raises(RuntimeError, match="optional nara_wpe"):
        offline_wpe_spectrum(spectrum, WPEConfig(taps=2), backend="nara_wpe")
