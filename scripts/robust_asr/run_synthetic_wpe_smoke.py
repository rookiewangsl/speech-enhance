#!/usr/bin/env python3
"""Run the NumPy WPE reference backend on generated multichannel signals."""

from __future__ import annotations

import argparse
import json

import numpy as np

from robust_asr.acoustics.rir import convolve_multichannel
from robust_asr.dereverb.wpe import WPEConfig, offline_wpe_waveform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def synthetic_rirs(sample_rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    length = round(0.35 * sample_rate)
    rirs = np.zeros((4, length), dtype=np.float64)
    for channel in range(4):
        direct = 24 + channel * 2
        rirs[channel, direct] = 1.0 - 0.04 * channel
        indices = np.arange(direct + 25, length, 37 + channel)
        decay = np.exp(-indices / (0.09 * sample_rate))
        rirs[channel, indices] += rng.normal(scale=0.25, size=len(indices)) * decay
    return rirs


def main() -> None:
    args = parse_args()
    if args.seconds <= 0:
        raise ValueError("seconds must be positive")
    sample_rate = 16_000
    samples = round(args.seconds * sample_rate)
    time = np.arange(samples) / sample_rate
    envelope = np.sin(np.pi * np.linspace(0.0, 1.0, samples)) ** 2
    rng = np.random.default_rng(args.seed + 1)
    clean = envelope * (
        0.12 * rng.normal(size=samples)
        + 0.2 * np.sin(2.0 * np.pi * 180.0 * time)
        + 0.15 * np.sin(2.0 * np.pi * 430.0 * time)
    )
    reverberant = convolve_multichannel(
        clean,
        synthetic_rirs(sample_rate, args.seed),
    ).signals
    config = WPEConfig(
        n_fft=256,
        win_length=256,
        hop_length=64,
        delay=2,
        taps=5,
        iterations=2,
    )
    output = offline_wpe_waveform(
        reverberant,
        config,
        backend="numpy_reference",
    )
    summary = {
        "backend": "numpy_reference_smoke_only",
        "input_shape": list(reverberant.shape),
        "output_shape": list(output.shape),
        "input_peak": float(np.max(np.abs(reverberant))),
        "output_peak": float(np.max(np.abs(output))),
        "finite": bool(np.all(np.isfinite(output))),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
