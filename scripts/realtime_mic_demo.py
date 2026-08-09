"""Capture and enhance microphone audio through native 48 kHz RNNoise."""

from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path

import numpy as np

from speech_frontend.audio import AudioData, write_audio
from speech_frontend.rnnoise import RNNoiseLibrary, StreamingRNNoise48k


def parse_device(value: str) -> int | str:
    """Accept either a numeric device index or a device-name substring."""

    try:
        return int(value)
    except ValueError:
        return value


def load_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "Microphone demo requires the optional demo dependency. "
            "Run: ./.venv/bin/pip install -e '.[demo]'"
        ) from error
    return sd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--device", type=parse_device)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--list-devices", action="store_true")
    arguments = parser.parse_args()
    if arguments.duration <= 0.0:
        raise ValueError("duration must be positive")

    sd = load_sounddevice()
    if arguments.list_devices:
        print(sd.query_devices())
        return

    sample_rate = 48_000
    block_samples = 480
    input_device = sd.query_devices(arguments.device, "input")
    input_device_name = str(input_device["name"])
    target_blocks = int(np.ceil(arguments.duration * 100))
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
    callback_status: list[str] = []
    dropped_blocks = 0

    def callback(indata, frames, _time_info, status) -> None:
        nonlocal dropped_blocks
        if status:
            callback_status.append(str(status))
        if frames != block_samples:
            callback_status.append(f"unexpected block size: {frames}")
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            dropped_blocks += 1

    library = RNNoiseLibrary(arguments.library)
    stream = StreamingRNNoise48k(library)
    captured: list[np.ndarray] = []
    enhanced: list[np.ndarray] = []
    vad: list[np.ndarray] = []
    processing_seconds = 0.0
    started = time.perf_counter()

    with sd.InputStream(
        samplerate=sample_rate,
        blocksize=block_samples,
        channels=1,
        dtype="float32",
        device=arguments.device,
        callback=callback,
    ):
        for block_index in range(target_blocks):
            try:
                block = audio_queue.get(timeout=2.0)
            except queue.Empty as error:
                raise RuntimeError(
                    "No microphone frames arrived within 2 seconds. "
                    "Check the selected input device and macOS microphone "
                    "permission."
                ) from error
            captured.append(block)
            processing_started = time.perf_counter()
            result = stream.process_chunk(block)
            processing_seconds += time.perf_counter() - processing_started
            enhanced.append(result.samples)
            vad.append(result.vad_probabilities)
            if (block_index + 1) % 50 == 0:
                probability = (
                    float(vad[-1][-1]) if vad[-1].size else 0.0
                )
                print(
                    f"{(block_index + 1) / 100:.1f}s "
                    f"VAD={probability:.3f}",
                    flush=True,
                )

    tail = stream.flush()
    enhanced.append(tail.samples)
    vad.append(tail.vad_probabilities)
    elapsed = time.perf_counter() - started
    input_audio = np.concatenate(captured)
    output_audio = np.concatenate(enhanced)
    probabilities = np.concatenate(vad)
    if output_audio.shape != input_audio.shape:
        raise RuntimeError("microphone demo input/output length mismatch")

    output_root = arguments.output_dir
    write_audio(
        output_root / "microphone_raw.wav",
        AudioData(input_audio.astype(np.float32), sample_rate),
    )
    write_audio(
        output_root / "microphone_rnnoise.wav",
        AudioData(output_audio.astype(np.float32), sample_rate),
    )
    report = {
        "input_device": input_device_name,
        "sample_rate": sample_rate,
        "block_samples": block_samples,
        "requested_duration_seconds": arguments.duration,
        "captured_samples": int(input_audio.size),
        "wall_time_seconds": elapsed,
        "processing_rtf": processing_seconds / arguments.duration,
        "algorithmic_delay_samples": stream.algorithmic_delay_samples,
        "algorithmic_delay_ms": (
            1_000 * stream.algorithmic_delay_samples / sample_rate
        ),
        "vad_frames": int(probabilities.size),
        "vad_mean": float(np.mean(probabilities)),
        "input_peak": float(np.max(np.abs(input_audio), initial=0.0)),
        "output_peak": float(np.max(np.abs(output_audio), initial=0.0)),
        "input_clipping_samples": int(
            np.count_nonzero(np.abs(input_audio) >= 1.0)
        ),
        "output_clipping_samples": int(
            np.count_nonzero(np.abs(output_audio) >= 1.0)
        ),
        "dropped_blocks": dropped_blocks,
        "callback_status": callback_status,
        "note": (
            "The saved enhanced file retains the physical 10 ms streaming "
            "delay; file playback comparison may compensate it offline."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
