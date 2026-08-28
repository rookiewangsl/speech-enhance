#!/usr/bin/env python3
"""Plot CER and direct-target signal-metric deltas against Raw as PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-summary", type=Path, required=True)
    parser.add_argument("--signal-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.suffix.lower() != ".png":
        raise ValueError("documentation figure output must be PNG")
    asr = json.loads(args.asr_summary.read_text(encoding="utf-8"))
    signal = json.loads(args.signal_summary.read_text(encoding="utf-8"))
    rt60 = tuple(map(float, signal["rt60_seconds"]))
    frontends = {
        "s_wpe_10": ("S-WPE-10", "#d97706", "s"),
        "s_wpe_40": ("S-WPE-40", "#7c3aed", "^"),
        "m_wpe_10": ("M-WPE-10", "#047857", "D"),
    }
    asr_values = {
        (str(row["frontend"]), float(row["target_rt60_seconds"])): float(
            row["cer"]
        )
        for row in asr["conditions"]
        if row["target_rt60_seconds"] is not None
    }
    signal_deltas = {
        (
            str(row["candidate"]),
            float(row["target_rt60_seconds"]),
        ): row["candidate_minus_raw"]
        for row in signal["paired_deltas"]
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 7.4),
        sharex=True,
        constrained_layout=True,
    )
    panels = (
        (axes[0], "cer", "Candidate − Raw CER (pp)"),
        (axes[1], "stoi", "Candidate − Raw STOI"),
        (axes[2], "si_sdr_db", "Candidate − Raw SI-SDR (dB)"),
    )
    for frontend, (label, color, marker) in frontends.items():
        for axis, metric, _ in panels:
            if metric == "cer":
                values = [
                    100.0
                    * (
                        asr_values[(frontend, value)]
                        - asr_values[("raw", value)]
                    )
                    for value in rt60
                ]
            else:
                values = [
                    float(signal_deltas[(frontend, value)][metric]["median"])
                    for value in rt60
                ]
            axis.plot(
                rt60,
                values,
                label=label,
                color=color,
                marker=marker,
                linewidth=2.1 if frontend == "m_wpe_10" else 1.7,
                markersize=5.5,
            )
    for axis, metric, ylabel in panels:
        axis.axhline(0, color="#111827", linewidth=1.0, linestyle=":")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
        if metric == "cer":
            axis.text(
                0.01,
                0.05,
                "lower is better",
                transform=axis.transAxes,
                color="#4b5563",
                fontsize=8.2,
            )
        else:
            axis.text(
                0.01,
                0.86,
                "higher is better",
                transform=axis.transAxes,
                color="#4b5563",
                fontsize=8.2,
            )
    axes[0].set_title("Frontend Metrics Do Not Fully Predict ASR Benefit")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[-1].set_xlabel("Target RT60 (s)")
    axes[-1].set_xticks(rt60)
    axes[-1].text(
        0.995,
        -0.27,
        "AISHELL-1 dev_frontend, 500 paired utterances",
        transform=axes[-1].transAxes,
        ha="right",
        color="#4b5563",
        fontsize=8.2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, format="png")
    plt.close(figure)


if __name__ == "__main__":
    main()
