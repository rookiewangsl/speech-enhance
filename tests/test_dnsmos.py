from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from speech_frontend.audio import AudioData
from speech_frontend.dnsmos import (
    calibrate,
    load_protocol,
    official_windows,
    score_audio,
)


ROOT = Path(__file__).parents[1]


class FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.inputs: list[np.ndarray] = []

    def run(self, output_names, input_feed):  # type: ignore[no-untyped-def]
        assert output_names == ["Identity:0"]
        self.inputs.append(input_feed["input_1"].copy())
        batch_size = input_feed["input_1"].shape[0]
        if self.output.shape[0] == 1:
            return [np.repeat(self.output, batch_size, axis=0)]
        assert self.output.shape[0] == batch_size
        return [self.output]


def protocol():  # type: ignore[no-untyped-def]
    config = json.loads((ROOT / "configs/dnsmos_p835.json").read_text(encoding="utf-8"))
    return load_protocol(config)


def test_official_windows_repeat_by_doubling_and_hop() -> None:
    samples = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    windows = official_windows(samples, input_samples=6, hop_samples=2)

    assert len(windows) == 2
    np.testing.assert_array_equal(windows[0], [0, 1, 2, 3, 0, 1])
    np.testing.assert_array_equal(windows[1], [2, 3, 0, 1, 2, 3])


def test_score_audio_matches_official_regular_calibration() -> None:
    current = protocol()
    session = FakeSession(np.asarray([[2.0, 3.0, 4.0]], dtype=np.float32))
    audio = AudioData(np.linspace(-0.25, 0.25, 16_000, dtype=np.float32), 16_000)

    result = score_audio(audio, session, current)

    assert result["num_hops"] == 7
    assert len(session.inputs) == 1
    assert session.inputs[0].shape == (7, 144_160)
    assert result["sig_raw"] == 2.0
    assert result["bak_raw"] == 3.0
    assert result["ovrl_raw"] == 4.0
    assert result["sig"] == pytest.approx(calibrate(2.0, current.calibration["sig"]))
    assert result["bak"] == pytest.approx(calibrate(3.0, current.calibration["bak"]))
    assert result["ovrl"] == pytest.approx(calibrate(4.0, current.calibration["ovrl"]))


def test_score_audio_rejects_unexpected_model_output() -> None:
    session = FakeSession(np.asarray([[2.0, 3.0]], dtype=np.float32))
    audio = AudioData(np.ones(16_000, dtype=np.float32), 16_000)

    with pytest.raises(ValueError, match="unexpected DNSMOS output"):
        score_audio(audio, session, protocol())


def test_score_audio_calibrates_each_hop_before_averaging() -> None:
    current = protocol()
    outputs = np.asarray(
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32
    )
    audio = AudioData(np.ones(160_000, dtype=np.float32), 16_000)
    shortened = type(current)(
        protocol_version=current.protocol_version,
        sample_rate=current.sample_rate,
        input_samples=144_000,
        hop_samples=16_000,
        input_name=current.input_name,
        output_name=current.output_name,
        model_sha256=current.model_sha256,
        inference_batch_size=current.inference_batch_size,
        calibration=current.calibration,
    )
    session = FakeSession(outputs)
    result = score_audio(audio, session, shortened)
    expected = np.mean(
        [calibrate(1.0, current.calibration["sig"]), calibrate(3.0, current.calibration["sig"])]
    )
    assert result["num_hops"] == 2
    assert result["sig_raw"] == 2.0
    assert result["sig"] == pytest.approx(expected)


def test_official_windows_reject_empty_audio() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        official_windows(np.empty(0, dtype=np.float32), input_samples=10, hop_samples=2)
