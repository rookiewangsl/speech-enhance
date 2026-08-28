#!/usr/bin/env python3
"""Render clean/direct-only Whisper input-audit results as PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.suffix.lower() != ".png":
        raise ValueError("documentation figure output must be PNG")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    by_condition = {
        str(row["condition"]): row for row in summary["conditions"]
    }
    order = (
        "clean_original",
        "clean_level",
        "direct_raw",
        "direct_s_wpe_10",
        "direct_s_wpe_40",
        "direct_m_wpe_10",
    )
    if set(order) - set(by_condition):
        raise ValueError("summary lacks one or more frozen audit conditions")
    labels = (
        "Clean\noriginal",
        "Clean\nlevel",
        "Direct\nraw",
        "Direct\nS-WPE-10",
        "Direct\nS-WPE-40",
        "Direct\nM-WPE-10",
    )
    colors = ("#64748b", "#94a3b8", "#374151", "#d97706", "#7c3aed", "#047857")
    cer = [100.0 * float(by_condition[name]["cer"]) for name in order]
    insertions = [int(by_condition[name]["insertions"]) for name in order]

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
    figure, (cer_axis, insertion_axis) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": (1.4, 1.0)},
        constrained_layout=True,
    )
    positions = list(range(len(order)))
    cer_axis.bar(positions, cer, color=colors, width=0.72)
    cer_axis.set_title("Frozen Whisper Input Audit Without Late Reverberation")
    cer_axis.set_ylabel("Corpus CER (%)")
    cer_axis.set_ylim(11.5, 14.7)
    cer_axis.grid(axis="y", alpha=0.22)
    for position, value in zip(positions, cer, strict=True):
        cer_axis.text(position, value + 0.08, f"{value:.2f}", ha="center", fontsize=8.5)

    insertion_axis.bar(positions, insertions, color=colors, width=0.72)
    insertion_axis.set_ylabel("Insertion errors")
    insertion_axis.set_xticks(positions, labels)
    insertion_axis.grid(axis="y", alpha=0.22)
    for position, value in zip(positions, insertions, strict=True):
        insertion_axis.text(position, value + 3, str(value), ha="center", fontsize=8.5)
    insertion_axis.text(
        0.995,
        -0.34,
        "AISHELL-1 dev_frontend, 500 paired utterances; all paired CER CIs cross zero",
        transform=insertion_axis.transAxes,
        ha="right",
        color="#4b5563",
        fontsize=8.2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, format="png")
    plt.close(figure)


if __name__ == "__main__":
    main()
