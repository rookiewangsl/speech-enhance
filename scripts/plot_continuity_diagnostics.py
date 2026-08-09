"""Visualize RNNoise speech intermittency and continuity-controller activity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from evaluate_rnnoise import enhance_in_chunks
from speech_frontend.audio import read_audio
from speech_frontend.rnnoise import (
    ContinuityConfig,
    RNNoiseLibrary,
    stabilize_output_continuity,
)


def frame_rms_db(samples: np.ndarray, frame_samples: int = 160) -> np.ndarray:
    values = []
    for offset in range(0, samples.size, frame_samples):
        frame = samples[offset : offset + frame_samples]
        values.append(20.0 * np.log10(np.sqrt(np.mean(frame**2) + 1e-10)))
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--noisy", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    arguments = parser.parse_args()

    clean = read_audio(arguments.clean)
    noisy = read_audio(arguments.noisy)
    if clean.sample_rate != 16_000 or noisy.sample_rate != 16_000:
        raise ValueError("continuity diagnostic requires 16 kHz audio")
    if clean.samples.shape != noisy.samples.shape:
        raise ValueError("clean and noisy lengths do not match")

    library = RNNoiseLibrary(arguments.library)
    r3, vad, _ = enhance_in_chunks(
        noisy.samples,
        library,
        chunk_size=137,
    )
    controlled = stabilize_output_continuity(
        r3,
        vad,
        config=ContinuityConfig(max_boost_db=3.0),
    )

    clean_rms = frame_rms_db(clean.samples)
    noisy_rms = frame_rms_db(noisy.samples)
    r3_rms = frame_rms_db(r3)
    time = np.arange(r3_rms.size) * 0.010
    vad_used = np.pad(
        controlled.vad_probability,
        (0, max(0, time.size - controlled.vad_probability.size)),
        mode="edge",
    )[: time.size]

    transform = ShortTimeFFT(
        hann(320, sym=False),
        hop=160,
        fs=16_000,
        mfft=512,
        scale_to="magnitude",
    )
    spectrum = np.abs(transform.stft(r3))
    spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-6))

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(12, 9),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(time, clean_rms, label="clean", linewidth=0.8)
    axes[0].plot(time, noisy_rms, label="noisy", linewidth=0.8)
    axes[0].plot(time, r3_rms, label="R3", linewidth=1.0)
    axes[0].set_ylabel("Frame RMS (dB)")
    axes[0].legend(loc="lower right", ncols=3)
    axes[0].grid(alpha=0.2)

    axes[1].step(
        time,
        vad_used,
        where="post",
        label="RNNoise VAD",
    )
    axes[1].axhline(0.60, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("VAD")
    axes[1].grid(alpha=0.2)

    axes[2].step(
        time,
        controlled.frame_gain_db[: time.size],
        where="post",
        color="#D55E00",
    )
    axes[2].set_ylabel("C1 boost (dB)")
    axes[2].set_ylim(-0.1, 3.2)
    axes[2].grid(alpha=0.2)

    extent = [
        0.0,
        r3.size / 16_000,
        0.0,
        8_000.0,
    ]
    axes[3].imshow(
        spectrum_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-75,
        vmax=-10,
        cmap="magma",
    )
    axes[3].set_ylabel("Frequency (Hz)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("R3 spectrogram")
    arguments.output_plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output_plot, dpi=170)
    plt.close(figure)

    report = {
        "clean": str(arguments.clean),
        "noisy": str(arguments.noisy),
        "frames": int(time.size),
        "vad_mean": float(np.mean(vad_used)),
        "boosted_frame_fraction": float(
            np.mean(controlled.frame_gain_db > 1e-4)
        ),
        "mean_boost_db": float(np.mean(controlled.frame_gain_db)),
        "maximum_boost_db": float(
            np.max(controlled.frame_gain_db, initial=0.0)
        ),
        "clipping_samples_before_limit": (
            controlled.clipping_samples_before_limit
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
