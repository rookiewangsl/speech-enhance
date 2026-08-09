"""Run the official RNNoise demo binary as a reproducible external baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from speech_frontend.audio import AudioData, read_audio, write_audio


RNNOISE_SAMPLE_RATE = 48_000
RNNOISE_FRAME_SAMPLES = 480


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if not arguments.binary.is_file():
        raise FileNotFoundError(f"RNNoise binary not found: {arguments.binary}")
    audio = read_audio(arguments.input)
    upsampled = resample_poly(
        audio.samples.astype(np.float64),
        RNNOISE_SAMPLE_RATE,
        audio.sample_rate,
    )
    original_48k_length = upsampled.size
    padding = (-original_48k_length) % RNNOISE_FRAME_SAMPLES
    padded = np.pad(upsampled, (0, padding))
    pcm = np.round(np.clip(padded, -1.0, 1.0) * 32767.0).astype("<i2")

    with tempfile.TemporaryDirectory(prefix="rnnoise_baseline_") as directory:
        temporary = Path(directory)
        input_raw = temporary / "input.raw"
        output_raw = temporary / "output.raw"
        pcm.tofile(input_raw)
        started = time.perf_counter()
        subprocess.run(
            [str(arguments.binary), str(input_raw), str(output_raw)],
            check=True,
            capture_output=True,
        )
        elapsed = time.perf_counter() - started
        enhanced_48k = (
            np.fromfile(output_raw, dtype="<i2").astype(np.float64) / 32767.0
        )

    enhanced_48k = enhanced_48k[:original_48k_length]
    enhanced = resample_poly(
        enhanced_48k,
        audio.sample_rate,
        RNNOISE_SAMPLE_RATE,
    )
    if enhanced.size < audio.samples.size:
        enhanced = np.pad(enhanced, (0, audio.samples.size - enhanced.size))
    enhanced = enhanced[: audio.samples.size]
    write_audio(
        arguments.output,
        AudioData(enhanced.astype(np.float32), audio.sample_rate),
    )
    duration = audio.samples.size / audio.sample_rate
    report = {
        "input": str(arguments.input),
        "output": str(arguments.output),
        "external_method": "official_rnnoise_pretrained",
        "input_sample_rate": audio.sample_rate,
        "model_sample_rate": RNNOISE_SAMPLE_RATE,
        "rtf_including_resampling": elapsed / duration,
        "peak": float(np.max(np.abs(enhanced))),
        "clipping_samples": int(np.count_nonzero(np.abs(enhanced) > 1.0)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
