#!/usr/bin/env python3
"""Plot shared-scale spectrograms for the p244_166 ASR failure case."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft


CONDITIONS = (
    ("Clean reference", "clean"),
    ("Noisy (babble, 5 dB)", "noisy"),
    ("MCRA + DD-Wiener", "mcra_dd_wiener"),
    ("RNNoise R3", "rnnoise_r3"),
)


def _read_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio: {path}")
    if np.issubdtype(audio.dtype, np.integer):
        scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
        audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite audio: {path}")
    return sample_rate, audio


def _spectrogram(sample_rate: int, audio: np.ndarray) -> tuple[np.ndarray, ...]:
    frequency, time, spectrum = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=400,
        noverlap=240,
        nfft=512,
        boundary=None,
        padded=False,
        scaling="spectrum",
    )
    magnitude_dbfs = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-8))
    return frequency, time, magnitude_dbfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--utterance-id", default="p244_166")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vmin", type=float, default=-100.0)
    parser.add_argument("--vmax", type=float, default=-20.0)
    args = parser.parse_args()

    if args.vmin >= args.vmax:
        raise ValueError("vmin must be less than vmax")

    panels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    expected_rate: int | None = None
    expected_samples: int | None = None
    for label, condition in CONDITIONS:
        path = args.audio_root / condition / f"{args.utterance_id}.wav"
        sample_rate, audio = _read_mono(path)
        if expected_rate is None:
            expected_rate = sample_rate
            expected_samples = len(audio)
        elif sample_rate != expected_rate or len(audio) != expected_samples:
            raise ValueError("all conditions must have identical sample rate and length")
        frequency, time, magnitude_dbfs = _spectrogram(sample_rate, audio)
        panels.append((label, frequency, time, magnitude_dbfs))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    figure, axes = plt.subplots(
        len(CONDITIONS),
        1,
        figsize=(10.5, 10.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, (label, frequency, time, magnitude_dbfs) in zip(axes, panels):
        image = axis.pcolormesh(
            time,
            frequency / 1000.0,
            magnitude_dbfs,
            shading="auto",
            cmap="magma",
            vmin=args.vmin,
            vmax=args.vmax,
            rasterized=True,
        )
        axis.set_title(label, loc="left", fontweight="bold")
        axis.set_ylabel("Frequency (kHz)")
        axis.set_ylim(0.0, 8.0)
        axis.set_yticks([0, 2, 4, 6, 8])
        axis.grid(False)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(
        f"{args.utterance_id}: clean, noisy, and enhanced spectrograms",
        fontsize=13,
        fontweight="bold",
    )
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, pad=0.015, aspect=35)
    colorbar.set_label("Magnitude (dBFS)")
    colorbar.set_ticks(np.arange(args.vmin, args.vmax + 1, 10.0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
