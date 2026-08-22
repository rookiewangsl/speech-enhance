from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from speech_frontend.rnnoise.resampler import StreamingDownsampler3
from speech_frontend.vad.endpoint import EndpointConfig, StreamingEndpointDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "realtime_asr_demo.py"
SPEC = importlib.util.spec_from_file_location("realtime_asr_demo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeSoundDevice:
    devices = [
        {
            "name": "Built-in Microphone",
            "max_input_channels": 1,
        },
        {
            "name": "Built-in Speakers",
            "max_input_channels": 0,
        },
        {
            "name": "USB Microphone",
            "max_input_channels": 2,
        },
    ]

    def query_devices(self, device=None, kind=None):
        if device is None:
            return self.devices
        if not isinstance(device, int) or device not in range(len(self.devices)):
            raise ValueError("unknown device")
        value = self.devices[device]
        if kind == "input" and value["max_input_channels"] == 0:
            raise ValueError("not an input")
        return value


def test_select_input_device_prompts_for_only_input_capable_devices() -> None:
    messages: list[str] = []

    device, details = MODULE.select_input_device(
        FakeSoundDevice(),
        None,
        input_fn=lambda _prompt: "2",
        output_fn=messages.append,
    )

    assert device == 2
    assert details["name"] == "USB Microphone"
    assert any("[0] Built-in Microphone" in message for message in messages)
    assert all("Speakers" not in message for message in messages)


def test_select_input_device_rejects_a_speaker_supplied_by_flag() -> None:
    with pytest.raises(ValueError, match="Not an input device"):
        MODULE.select_input_device(FakeSoundDevice(), 1)


def test_live_asr_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["realtime_asr_demo.py", "--output-dir", "outputs/live_demo"],
    )

    arguments = MODULE.parse_arguments()

    assert arguments.asr_device == "cpu"


def test_milliseconds_to_frames_rounds_up_and_rejects_nonpositive_values() -> None:
    assert MODULE.milliseconds_to_frames(20.0, "onset-ms") == 2
    assert MODULE.milliseconds_to_frames(21.0, "onset-ms") == 3
    with pytest.raises(ValueError, match="onset-ms"):
        MODULE.milliseconds_to_frames(0.0, "onset-ms")


def test_process_enhanced_frames_emits_completed_16khz_segment() -> None:
    endpoint = StreamingEndpointDetector(
        EndpointConfig(
            onset_frames=1,
            hangover_frames=1,
            pre_roll_frames=1,
            minimum_speech_frames=1,
            maximum_segment_frames=10,
        )
    )
    downsampler = StreamingDownsampler3()

    first = MODULE.process_enhanced_frames(
        np.full(480, 0.1, dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        downsampler,
        endpoint,
    )
    second = MODULE.process_enhanced_frames(
        np.zeros(480, dtype=np.float32),
        np.array([0.1], dtype=np.float32),
        downsampler,
        endpoint,
    )

    assert first == []
    assert len(second) == 1
    assert second[0].samples.size == 320
    assert second[0].reason == "silence"


def test_process_enhanced_frames_rejects_unmatched_vad_scores() -> None:
    with pytest.raises(RuntimeError, match="frame mismatch"):
        MODULE.process_enhanced_frames(
            np.zeros(480, dtype=np.float32),
            np.array([0.5, 0.5], dtype=np.float32),
            StreamingDownsampler3(),
            StreamingEndpointDetector(),
        )
