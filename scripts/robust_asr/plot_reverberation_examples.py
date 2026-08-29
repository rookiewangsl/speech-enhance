#!/usr/bin/env python3
"""Plot controlled Clean/Raw/M-WPE spectrograms and RIR decay as PNG."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import stft

from robust_asr.acoustics.rir import convolve_multichannel
from robust_asr.dereverb.frontend import apply_frontend
from robust_asr.manifest import read_jsonl
from robust_asr.paths import require_data_root


@dataclass(frozen=True)
class Example:
    target_rt60: float
    measured_rt60: float
    drr_db: float
    arrival_sample: int
    full_rir: np.ndarray
    clean_timeline: np.ndarray
    raw: np.ndarray
    m_wpe: np.ndarray


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--utterance-id", default="BAC009S0913W0222"
    )
    parser.add_argument("--rir-family-id", default="test_r000_f000")
    parser.add_argument("--rt60", type=float, nargs="+", default=(0.2, 0.6, 1.0))
    parser.add_argument("--spectrogram-output", type=Path, required=True)
    parser.add_argument("--rir-output", type=Path, required=True)
    return parser.parse_args()


def _one(rows: list[dict[str, Any]], **identity: Any) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in identity.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {identity}, got {len(selected)}")
    return selected[0]


def _first_float(value: Any) -> float:
    if isinstance(value, list):
        value = value[0]
    return float(value)


def _rms_match(signal: np.ndarray, target_dbfs: float = -25.0) -> np.ndarray:
    output = np.asarray(signal, dtype=np.float64).copy()
    rms = float(np.sqrt(np.mean(output**2)))
    if rms <= np.finfo(float).tiny:
        raise ValueError("cannot RMS-match silent clean speech")
    output *= 10.0 ** (target_dbfs / 20.0) / rms
    peak_limit = 10.0 ** (-1.0 / 20.0)
    peak = float(np.max(np.abs(output)))
    if peak > peak_limit:
        output *= peak_limit / peak
    return output


def _load_examples(
    *,
    root: Path,
    utterance_id: str,
    rir_family_id: str,
    rt60_values: tuple[float, ...],
) -> tuple[dict[str, Any], list[Example], int]:
    utterances = read_jsonl(
        root / "manifests" / "aishell1" / "aishell1_test_reverb.jsonl"
    )
    utterance = _one(utterances, utterance_id=utterance_id)
    audio_path = root / "corpora" / "aishell1" / str(utterance["audio_path"])
    clean, sample_rate = sf.read(audio_path, dtype="float64", always_2d=False)
    if clean.ndim != 1 or sample_rate != 16_000:
        raise ValueError("example speech must be mono 16 kHz")
    clean_scaled = _rms_match(clean)

    rir_rows = read_jsonl(root / "rir" / "pyroom_v1" / "test.jsonl")
    family = [row for row in rir_rows if row["rir_family_id"] == rir_family_id]
    examples: list[Example] = []
    for target in rt60_values:
        candidates = [
            row
            for row in family
            if abs(float(row["target_rt60_seconds"]) - target) < 1e-8
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"expected one {rir_family_id} RIR at RT60={target}"
            )
        rir_row = candidates[0]
        rir_path = root / "rir" / "pyroom_v1" / str(rir_row["path"])
        with np.load(rir_path) as arrays:
            full = np.asarray(arrays["full"], dtype=np.float64)
            direct = np.asarray(arrays["direct"], dtype=np.float64)
        result = convolve_multichannel(clean, full)
        raw = apply_frontend(result.signals, "raw")
        m_wpe = apply_frontend(result.signals, "m_wpe_10")
        arrival = int(np.argmax(np.abs(direct[0])))
        clean_timeline = np.zeros(result.signals.shape[1], dtype=np.float64)
        end = min(arrival + clean_scaled.size, clean_timeline.size)
        clean_timeline[arrival:end] = clean_scaled[: end - arrival]
        examples.append(
            Example(
                target_rt60=target,
                measured_rt60=_first_float(rir_row["measured_rt60_seconds"]),
                drr_db=_first_float(rir_row["drr_db"]),
                arrival_sample=arrival,
                full_rir=full[0],
                clean_timeline=clean_timeline,
                raw=np.asarray(raw, dtype=np.float64),
                m_wpe=np.asarray(m_wpe, dtype=np.float64),
            )
        )
    return utterance, examples, sample_rate


def _spectrogram(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, ...]:
    frequencies, times, spectrum = stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=400,
        noverlap=240,
        nfft=512,
        boundary=None,
        padded=False,
    )
    magnitude_db = 20.0 * np.log10(
        np.maximum(np.abs(spectrum), np.finfo(float).tiny)
    )
    return frequencies, times, magnitude_db


def _plot_spectrograms(
    *,
    utterance: dict[str, Any],
    examples: list[Example],
    sample_rate: int,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = (
        ("clean_timeline", "Clean (RMS matched)"),
        ("raw", "Raw reverberant"),
        ("m_wpe", "4-channel M-WPE-10"),
    )
    longest = max(
        signal.size
        for example in examples
        for signal in (example.clean_timeline, example.raw, example.m_wpe)
    )
    prepared: dict[tuple[int, str], tuple[np.ndarray, ...]] = {}
    maxima: list[float] = []
    for row_index, example in enumerate(examples):
        for field, _ in columns:
            signal = np.asarray(getattr(example, field), dtype=np.float64)
            padded = np.pad(signal, (0, longest - signal.size))
            values = _spectrogram(padded, sample_rate)
            prepared[(row_index, field)] = values
            maxima.append(float(np.max(values[2])))
    color_max = max(maxima)
    color_min = color_max - 75.0

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        len(examples),
        len(columns),
        figsize=(12.4, 8.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for row_index, example in enumerate(examples):
        source_end = (
            example.arrival_sample + int(utterance["frames"])
        ) / sample_rate
        for column_index, (field, title) in enumerate(columns):
            axis = axes[row_index, column_index]
            frequencies, times, magnitude_db = prepared[(row_index, field)]
            image = axis.pcolormesh(
                times,
                frequencies / 1000.0,
                magnitude_db,
                shading="auto",
                cmap="magma",
                vmin=color_min,
                vmax=color_max,
            )
            axis.axvline(source_end, color="white", linestyle="--", linewidth=1.0)
            axis.set_ylim(0.0, 8.0)
            if row_index == 0:
                axis.set_title(title)
            if column_index == 0:
                axis.set_ylabel(
                    f"RT60 {example.target_rt60:.1f} s\n"
                    f"DRR {example.drr_db:+.1f} dB\nFrequency (kHz)"
                )
            if row_index == len(examples) - 1:
                axis.set_xlabel("Time (s)")
    if image is None:
        raise RuntimeError("no spectrograms were rendered")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.92, pad=0.01)
    colorbar.set_label("STFT magnitude (dB, shared scale)")
    figure.suptitle(
        f"Controlled Reverberation Example | {utterance['utterance_id']} | "
        f"source {float(utterance['duration_seconds']):.2f} s\n"
        "Dashed line: clean source end; energy to the right is the reverberant tail"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, format="png")
    plt.close(figure)


def _rir_envelope_db(rir: np.ndarray, window_samples: int) -> np.ndarray:
    energy = np.asarray(rir, dtype=np.float64) ** 2
    smoothed = np.convolve(
        energy, np.ones(window_samples) / window_samples, mode="same"
    )
    return 10.0 * np.log10(np.maximum(smoothed, np.finfo(float).tiny))


def _plot_rir_decay(
    *, examples: list[Example], sample_rate: int, output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#2563eb", "#d97706", "#047857")
    global_reference = max(float(np.max(np.abs(item.full_rir))) for item in examples)
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    maximum_time = 0.0
    for example, color in zip(examples, colors, strict=True):
        start = example.arrival_sample
        rir = example.full_rir[start:]
        times = np.arange(rir.size) / sample_rate
        maximum_time = max(maximum_time, float(times[-1]))
        envelope_db = _rir_envelope_db(rir, max(1, round(0.005 * sample_rate)))
        envelope_db -= 20.0 * np.log10(global_reference)
        energy = rir**2
        decay = np.cumsum(energy[::-1])[::-1]
        decay_db = 10.0 * np.log10(
            np.maximum(decay / max(float(decay[0]), np.finfo(float).tiny), 1e-10)
        )
        label = (
            f"target {example.target_rt60:.1f} s | "
            f"measured {example.measured_rt60:.2f} s | DRR {example.drr_db:+.1f} dB"
        )
        axes[0].plot(times, envelope_db, color=color, linewidth=1.4, label=label)
        axes[1].plot(times, decay_db, color=color, linewidth=2.0, label=label)
        axes[1].scatter(
            [example.measured_rt60], [-60.0], color=color, marker="o", s=25, zorder=3
        )
    axes[0].set_title("Reference-channel RIR envelope")
    axes[0].set_xlabel("Time after direct arrival (s)")
    axes[0].set_ylabel("Smoothed energy (dB, shared reference)")
    axes[1].set_title("Schroeder energy decay")
    axes[1].set_xlabel("Time after direct arrival (s)")
    axes[1].set_ylabel("Remaining energy (dB)")
    for axis in axes:
        axis.set_xlim(0.0, maximum_time)
        axis.set_ylim(-80.0, 3.0)
        axis.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8.3, loc="upper right")
    figure.suptitle("One Test Geometry Across Three Reverberation Times")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, format="png")
    plt.close(figure)


def main() -> None:
    args = arguments()
    for output in (args.spectrogram_output, args.rir_output):
        if output.suffix.lower() != ".png":
            raise ValueError("documentation figure output must be PNG")
    root = require_data_root(args.data_root)
    utterance, examples, sample_rate = _load_examples(
        root=root,
        utterance_id=args.utterance_id,
        rir_family_id=args.rir_family_id,
        rt60_values=tuple(args.rt60),
    )
    _plot_spectrograms(
        utterance=utterance,
        examples=examples,
        sample_rate=sample_rate,
        output=args.spectrogram_output,
    )
    _plot_rir_decay(examples=examples, sample_rate=sample_rate, output=args.rir_output)
    print(
        f"Rendered {utterance['utterance_id']} with {args.rir_family_id}: "
        f"{args.spectrogram_output}, {args.rir_output}"
    )


if __name__ == "__main__":
    main()
