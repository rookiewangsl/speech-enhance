#!/usr/bin/env python3
"""Render the frozen Whisper Raw/WPE development CER curve as PNG."""

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
    conditions = summary["conditions"]
    clean_rows = [row for row in conditions if row["frontend"] == "clean"]
    if len(clean_rows) != 1:
        raise ValueError("summary must contain exactly one clean condition")
    clean_cer = 100.0 * float(clean_rows[0]["cer"])
    frontends = {
        "raw": ("Raw", "#6b7280", "o", "-"),
        "s_wpe_10": ("S-WPE-10", "#d97706", "s", "--"),
        "s_wpe_40": ("S-WPE-40", "#7c3aed", "^", "--"),
        "m_wpe_10": ("M-WPE-10", "#047857", "D", "-"),
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for frontend, (label, color, marker, linestyle) in frontends.items():
        rows = sorted(
            (row for row in conditions if row["frontend"] == frontend),
            key=lambda row: float(row["target_rt60_seconds"]),
        )
        if not rows:
            raise ValueError(f"summary has no {frontend} rows")
        axis.plot(
            [float(row["target_rt60_seconds"]) for row in rows],
            [100.0 * float(row["cer"]) for row in rows],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2 if frontend == "m_wpe_10" else 1.7,
            markersize=5.5,
        )
    axis.axhline(
        clean_cer,
        color="#111827",
        linestyle=":",
        linewidth=1.4,
        label=f"Clean ({clean_cer:.2f}%)",
    )
    axis.set_title("Frozen Whisper-small on Simulated Reverberation")
    axis.set_xlabel("Target RT60 (s)")
    axis.set_ylabel("Normalized corpus CER (%)")
    axis.set_xticks(summary["rt60_seconds"])
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2)
    axis.text(
        0.985,
        0.04,
        "AISHELL-1 dev_frontend, 500 utterances",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#4b5563",
        fontsize=8.5,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, format="png")
    plt.close(figure)


if __name__ == "__main__":
    main()
