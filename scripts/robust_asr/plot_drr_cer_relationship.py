#!/usr/bin/env python3
"""Plot RIR-level DRR against W0 Raw CER degradation on held-out test."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr, theilslopes

from robust_asr.manifest import read_jsonl


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rt60", type=float, nargs="+", default=(0.6, 0.8, 1.0))
    parser.add_argument("--y-max", type=float, default=45.0)
    return parser.parse_args()


def _points(
    rows: list[dict[str, Any]], rt60: float
) -> tuple[np.ndarray, np.ndarray, float]:
    clean = {
        str(row["utterance_id"]): float(row["cer"])
        for row in rows
        if row.get("frontend") == "clean"
    }
    selected = [
        row
        for row in rows
        if row.get("frontend") == "raw"
        and row.get("target_rt60_seconds") is not None
        and abs(float(row["target_rt60_seconds"]) - rt60) < 1e-8
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row["rir_id"]), []).append(row)
    x: list[float] = []
    y: list[float] = []
    for values in grouped.values():
        x.append(float(values[0]["reference_drr_db"]))
        y.append(
            100.0
            * float(
                np.mean(
                    [
                        float(row["cer"]) - clean[str(row["utterance_id"])]
                        for row in values
                    ]
                )
            )
        )
    if len(x) < 3:
        raise ValueError(f"not enough RIR groups for RT60={rt60}")
    rho = float(spearmanr(x, y).statistic)
    return np.asarray(x), np.asarray(y), rho


def main() -> None:
    args = arguments()
    if args.output.suffix.lower() != ".png":
        raise ValueError("documentation figure output must be PNG")
    rows = read_jsonl(args.results)
    series = [(_points(rows, value), value) for value in args.rt60]
    all_x = np.concatenate([item[0][0] for item in series])
    all_y = np.concatenate([item[0][1] for item in series])
    x_padding = 0.08 * max(float(np.ptp(all_x)), 1.0)
    y_padding = 0.08 * max(float(np.ptp(all_y[all_y <= args.y_max])), 1.0)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#2563eb", "#d97706", "#047857")
    figure, axes = plt.subplots(
        1, len(series), figsize=(12.0, 3.9), sharex=True, sharey=True, constrained_layout=True
    )
    if len(series) == 1:
        axes = [axes]
    for axis, ((x, y, rho), rt60), color in zip(
        axes, series, colors, strict=False
    ):
        visible = y <= args.y_max
        axis.scatter(
            x[visible], y[visible], s=30, alpha=0.72, color=color, edgecolors="none"
        )
        clipped_y = args.y_max - 1.0
        for outlier_x, outlier_y in zip(x[~visible], y[~visible], strict=True):
            axis.scatter(
                [outlier_x],
                [clipped_y],
                s=48,
                color=color,
                marker="^",
                edgecolors="none",
            )
            axis.annotate(
                f"{outlier_y:.1f} pp",
                (outlier_x, clipped_y),
                xytext=(0, -13),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8.3,
            )
        slope, intercept, _, _ = theilslopes(y, x)
        guide_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        axis.plot(
            guide_x,
            intercept + slope * guide_x,
            color=color,
            linewidth=1.7,
        )
        axis.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.0)
        axis.set_title(f"RT60 = {rt60:.1f} s | Spearman rho = {rho:+.3f}")
        axis.set_xlabel("Reference-channel DRR (dB)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean CER degradation vs paired clean (pp)")
    axes[0].set_xlim(float(np.min(all_x) - x_padding), float(np.max(all_x) + x_padding))
    axes[0].set_ylim(float(np.min(all_y) - y_padding), args.y_max)
    figure.suptitle("DRR vs W0 CER Degradation within Fixed RT60")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, format="png")
    plt.close(figure)


if __name__ == "__main__":
    main()
