"""Plot input waveform and per-frame RNNoise VAD probability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly

from speech_frontend.audio import read_audio
from speech_frontend.rnnoise import RNNOISE_SAMPLE_RATE, RNNoiseLibrary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    arguments = parser.parse_args()

    audio = read_audio(arguments.input)
    if audio.sample_rate == RNNOISE_SAMPLE_RATE:
        model_input = audio.samples.astype(np.float64)
    else:
        model_input = resample_poly(
            audio.samples.astype(np.float64),
            RNNOISE_SAMPLE_RATE,
            audio.sample_rate,
        )
    model_input = np.asarray(
        np.clip(model_input, -1.0, 1.0),
        dtype=np.float32,
    )

    library = RNNoiseLibrary(arguments.library)
    with library.create_state() as state:
        result = state.process_audio(model_input)

    frame_time = (
        np.arange(result.vad_probabilities.size, dtype=np.float64) * 0.010
    )
    waveform_time = (
        np.arange(model_input.size, dtype=np.float64) / RNNOISE_SAMPLE_RATE
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 5),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(waveform_time, model_input, linewidth=0.5)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"RNNoise diagnostic: {arguments.input.name}")
    axes[0].grid(alpha=0.2)
    axes[1].step(
        frame_time,
        result.vad_probabilities,
        where="post",
        linewidth=1.0,
    )
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("VAD probability")
    axes[1].grid(alpha=0.2)
    arguments.output_plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output_plot, dpi=160)
    plt.close(figure)

    report = {
        "input": str(arguments.input),
        "input_sample_rate": audio.sample_rate,
        "model_sample_rate": RNNOISE_SAMPLE_RATE,
        "model_samples": int(model_input.size),
        "frames": int(result.vad_probabilities.size),
        "padding_samples_48k": result.padding_samples,
        "vad_min": float(np.min(result.vad_probabilities, initial=1.0)),
        "vad_max": float(np.max(result.vad_probabilities, initial=0.0)),
        "vad_mean": float(np.mean(result.vad_probabilities)),
        "vad_timeline_note": (
            "VAD belongs to the current input frame; enhanced audio returned "
            "by the same API call belongs to the one-frame-delayed spectrum."
        ),
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
