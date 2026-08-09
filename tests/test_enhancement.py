from __future__ import annotations

import numpy as np
import pytest

from speech_frontend.enhancement.spectral_subtraction import (
    oracle_wiener_spectrum,
    spectral_subtraction,
)
from speech_frontend.enhancement.wiener import (
    DecisionDirectedWiener,
    DualUncertaintyWiener,
    WienerConfig,
    instantaneous_wiener_gain,
)
from speech_frontend.enhancement.om_lsa import OMLSA, OMLSAConfig
from speech_frontend.metrics import si_sdr, stoi


def test_spectral_subtraction_preserves_phase_and_gain_bounds() -> None:
    spectrum = np.array([1 + 1j, -2j, 0.5 - 0.25j])
    noise_psd = np.array([0.5, 2.0, 10.0])

    enhanced = spectral_subtraction(
        spectrum,
        noise_psd,
        gain_floor=0.1,
    )
    gain = np.divide(
        np.abs(enhanced),
        np.abs(spectrum),
        out=np.zeros(3),
        where=np.abs(spectrum) > 0,
    )

    assert np.all((0.1 <= gain) & (gain <= 1.0))
    np.testing.assert_allclose(np.angle(enhanced), np.angle(spectrum))


def test_oracle_wiener_is_identity_when_no_noise_is_present() -> None:
    clean = np.array([1 + 2j, 3 - 4j])

    enhanced = oracle_wiener_spectrum(clean, clean)

    np.testing.assert_allclose(enhanced, clean, atol=1e-12)


def test_instantaneous_wiener_gain_is_finite_and_bounded() -> None:
    gain = instantaneous_wiener_gain(
        np.array([0.0, 1.0, 10.0, 1e20]),
        np.array([0.0, 2.0, 1.0, 1e-20]),
        gain_floor=0.05,
    )

    assert np.all(np.isfinite(gain))
    assert np.all((0.05 <= gain) & (gain <= 1.0))


def test_decision_directed_wiener_has_state_and_reset() -> None:
    processor = DecisionDirectedWiener(
        WienerConfig(
            alpha_dd=0.96,
            gain_decrease_smoothing=0.7,
            gain_increase_smoothing=0.4,
        )
    )
    spectrum = np.ones(5, dtype=np.complex128)
    noise = np.ones(5)

    _, first_gain = processor.process_frame(spectrum, noise)
    _, second_gain = processor.process_frame(spectrum, noise)
    processor.reset()
    _, reset_gain = processor.process_frame(spectrum, noise)

    assert np.all((0.05 <= second_gain) & (second_gain <= 1.0))
    assert not np.allclose(second_gain, first_gain)
    np.testing.assert_allclose(reset_gain, first_gain)


def test_decision_directed_wiener_rejects_bin_count_change() -> None:
    processor = DecisionDirectedWiener()
    processor.process_frame(np.ones(3), np.ones(3))

    with pytest.raises(ValueError, match="frequency-bin"):
        processor.process_frame(np.ones(4), np.ones(4))


def test_frequency_smoothing_reduces_isolated_gain_notches() -> None:
    spectrum = np.array([10.0, 10.0, 0.1, 10.0, 10.0], dtype=np.complex128)
    noise = np.ones(5)
    base = DecisionDirectedWiener(
        WienerConfig(
            alpha_dd=0.0,
            gain_floor=0.05,
            gain_decrease_smoothing=0.0,
            gain_increase_smoothing=0.0,
        )
    )
    smooth = DecisionDirectedWiener(
        WienerConfig(
            alpha_dd=0.0,
            gain_floor=0.05,
            gain_decrease_smoothing=0.0,
            gain_increase_smoothing=0.0,
            gain_frequency_smoothing=0.9,
        )
    )

    _, base_gain = base.process_frame(spectrum, noise)
    _, smooth_gain = smooth.process_frame(spectrum, noise)

    assert np.max(np.abs(np.diff(smooth_gain))) < np.max(np.abs(np.diff(base_gain)))
    assert np.all((0.05 <= smooth_gain) & (smooth_gain <= 1.0))


def test_startup_ramp_defers_unreliable_first_noise_estimate() -> None:
    processor = DecisionDirectedWiener(
        WienerConfig(
            alpha_dd=0.0,
            gain_floor=0.1,
            gain_decrease_smoothing=0.0,
            gain_increase_smoothing=0.0,
            startup_frames=4,
        )
    )
    spectrum = np.ones(3, dtype=np.complex128)
    noise = np.ones(3)

    gains = [processor.process_frame(spectrum, noise)[1] for _ in range(5)]

    np.testing.assert_allclose(gains[0], 1.0)
    assert np.all(gains[1] < gains[0])
    np.testing.assert_allclose(gains[-1], 0.1)


def test_om_lsa_returns_bounded_gain_and_speech_probability() -> None:
    processor = OMLSA(OMLSAConfig(gain_floor=0.1))

    enhanced, gain, probability = processor.process_frame(
        np.array([1.0, 5.0, 0.2], dtype=np.complex128),
        np.array([1.0, 1.0, 1.0]),
        local_speech_presence_probability=np.array([0.0, 0.7, 1.0]),
    )

    assert enhanced.shape == gain.shape == probability.shape == (3,)
    assert np.all(np.isfinite(enhanced))
    assert np.all((0.1 <= gain) & (gain <= 1.0))
    assert np.all((0.0 <= probability) & (probability <= 1.0))


def test_om_lsa_reset_restores_first_frame_behavior() -> None:
    processor = OMLSA()
    spectrum = np.full(4, 2.0, dtype=np.complex128)
    noise = np.ones(4)
    presence = np.full(4, 0.5)

    _, first_gain, _ = processor.process_frame(
        spectrum,
        noise,
        local_speech_presence_probability=presence,
    )
    processor.process_frame(
        spectrum,
        noise,
        local_speech_presence_probability=presence,
    )
    processor.reset()
    _, reset_gain, _ = processor.process_frame(
        spectrum,
        noise,
        local_speech_presence_probability=presence,
    )

    np.testing.assert_allclose(reset_gain, first_gain)


def test_dual_uncertainty_wiener_returns_bounded_gain_and_probability() -> None:
    processor = DualUncertaintyWiener()

    enhanced, gain, probability = processor.process_frame(
        np.ones(7, dtype=np.complex128),
        np.full(7, 0.2),
        vad_speech_probability=0.8,
    )

    assert enhanced.shape == gain.shape == probability.shape == (7,)
    assert np.all((0.05 <= gain) & (gain <= 1.0))
    assert np.all((0.0 <= probability) & (probability <= 1.0))


def test_dual_uncertainty_wiener_rejects_invalid_vad_prior_strength() -> None:
    with pytest.raises(ValueError, match="vad_prior_strength"):
        WienerConfig(vad_prior_strength=1.1)


def test_si_sdr_is_high_for_identical_signal_and_lower_for_noise() -> None:
    reference = np.array([0.1, -0.4, 0.3, 0.2, -0.1])

    identical = si_sdr(reference, reference)
    noisy = si_sdr(reference, reference + np.array([0.1, 0.1, 0.0, -0.1, 0.0]))

    assert identical > 100.0
    assert noisy < identical


def test_stoi_is_high_for_identical_speech_like_signal() -> None:
    sample_rate = 16_000
    time = np.arange(2 * sample_rate) / sample_rate
    signal = 0.2 * np.sin(2 * np.pi * 180 * time)

    score = stoi(signal, signal, sample_rate=sample_rate)

    assert score > 0.99
