from __future__ import annotations

import numpy as np

from speech_frontend.vad.energy import EnergyVAD, EnergyVADConfig
from speech_frontend.vad.state_machine import (
    VADStateConfig,
    VADStateMachine,
)
from speech_frontend.vad.statistical import StatisticalVAD
from speech_frontend.vad.external import WebRTCVAD


def test_state_machine_applies_pre_roll_and_hangover() -> None:
    scores = np.array([0, 0, 9, 10, 7, 3, 2, 1, 0], dtype=np.float64)
    machine = VADStateMachine(
        VADStateConfig(
            threshold_on=8,
            threshold_off=4,
            onset_frames=2,
            pre_roll_frames=1,
            hangover_frames=2,
            minimum_speech_frames=1,
        )
    )

    labels = machine.apply(scores)

    np.testing.assert_array_equal(
        labels,
        [False, True, True, True, True, True, True, False, False],
    )


def test_energy_vad_detects_loud_region_with_context() -> None:
    sample_rate = 16_000
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=0.001, size=sample_rate)
    time = np.arange(sample_rate) / sample_rate
    speech_like = 0.1 * np.sin(2 * np.pi * 220 * time)
    signal = np.concatenate((noise, speech_like, noise))
    vad = EnergyVAD()

    result = vad.detect(signal)

    assert result.speech_probability.shape == result.frame_labels.shape
    assert result.frame_labels[100:200].mean() > 0.8
    assert result.frame_labels[:50].mean() < 0.2
    assert len(result.segments) == 1
    start, stop = result.segments[0]
    assert start < sample_rate
    assert stop > 2 * sample_rate


def test_energy_vad_handles_speech_at_file_start_without_calibration() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate) / sample_rate
    signal = 0.1 * np.sin(2 * np.pi * 180 * time)
    vad = EnergyVAD(
        EnergyVADConfig(
            pre_roll_ms=0,
            minimum_speech_ms=20,
        )
    )

    result = vad.detect(signal)

    assert result.frame_labels[:20].mean() > 0.8
    assert result.segments[0][0] == 0


def test_energy_vad_silence_is_not_speech() -> None:
    result = EnergyVAD().detect(np.zeros(16_000))

    assert not result.frame_labels.any()
    assert result.segments == ()
    assert np.all(np.isfinite(result.speech_probability))


def test_statistical_vad_silence_is_not_speech() -> None:
    result = StatisticalVAD().detect(np.zeros(16_000))

    assert not result.frame_labels.any()
    assert result.segments == ()
    assert np.all(np.isfinite(result.speech_probability))


def test_webrtc_vad_silence_is_not_speech() -> None:
    result = WebRTCVAD().detect(np.zeros(16_000))

    assert not result.frame_labels.any()
    assert result.segments == ()
