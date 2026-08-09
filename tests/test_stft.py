from __future__ import annotations

import numpy as np
import pytest

from speech_frontend.stft import STFT, STFTConfig, STFTResult


@pytest.mark.parametrize("length", [1, 159, 160, 319, 320, 321, 16_003])
def test_stft_round_trip_random_signal(length: int) -> None:
    rng = np.random.default_rng(20260724 + length)
    signal = rng.normal(size=length)
    transform = STFT()

    reconstructed = transform.synthesize(transform.analyze(signal))

    assert reconstructed.shape == signal.shape
    np.testing.assert_allclose(reconstructed, signal, atol=1e-10, rtol=1e-10)


def test_stft_round_trip_impulse_at_both_boundaries() -> None:
    signal = np.zeros(1_001, dtype=np.float64)
    signal[0] = 1.0
    signal[-1] = -0.5
    transform = STFT()

    reconstructed = transform.synthesize(transform.analyze(signal))

    np.testing.assert_allclose(reconstructed, signal, atol=1e-10, rtol=1e-10)


def test_stft_has_expected_frequency_bin_count() -> None:
    transform = STFT(STFTConfig(320, 160, 512))

    result = transform.analyze(np.zeros(1_600))

    assert result.spectrum.shape[1] == 257


def test_default_enhancement_configuration_uses_75_percent_overlap() -> None:
    config = STFTConfig()

    assert config.frame_length == 512
    assert config.hop_length == 128
    assert config.fft_length == 512


def test_stft_rejects_hop_equal_to_frame_length() -> None:
    with pytest.raises(ValueError, match="hop_length"):
        STFTConfig(frame_length=320, hop_length=320, fft_length=512)


def test_stft_rejects_non_finite_input() -> None:
    transform = STFT()

    with pytest.raises(ValueError, match="NaN"):
        transform.analyze(np.array([0.0, np.nan]))


def test_synthesis_rejects_wrong_frequency_bin_count() -> None:
    transform = STFT()
    bad_result = STFTResult(
        spectrum=np.zeros((4, 10), dtype=np.complex128),
        original_length=320,
        left_padding=160,
    )

    with pytest.raises(ValueError, match="frequency-bin"):
        transform.synthesize(bad_result)


def test_synthesis_rejects_non_finite_spectrum() -> None:
    transform = STFT()
    result = transform.analyze(np.zeros(1_000))
    result.spectrum[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        transform.synthesize(result)


def test_synthesis_rejects_inconsistent_metadata() -> None:
    transform = STFT()
    result = transform.analyze(np.zeros(1_000))
    bad_result = STFTResult(
        spectrum=result.spectrum,
        original_length=result.original_length,
        left_padding=0,
    )

    with pytest.raises(ValueError, match="left_padding"):
        transform.synthesize(bad_result)
