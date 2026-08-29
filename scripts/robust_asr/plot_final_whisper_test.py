#!/usr/bin/env python3
"""Render final Whisper test CER and error-type figures as PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = {
    "w0": ("W0 pretrained", "#6b7280"),
    "clean": ("Clean LoRA", "#2563eb"),
    "mct": ("MCT LoRA", "#047857"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--w0-summary", type=Path, required=True)
    parser.add_argument("--clean-summary", type=Path, required=True)
    parser.add_argument("--mct-summary", type=Path, required=True)
    parser.add_argument("--cer-output", type=Path, required=True)
    parser.add_argument("--error-output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _condition(
    summary: dict[str, Any], *, frontend: str, rt60: float | None
) -> dict[str, Any]:
    selected = [
        row
        for row in summary["conditions"]
        if row["frontend"] == frontend
        and (
            row["target_rt60_seconds"] is None
            if rt60 is None
            else abs(float(row["target_rt60_seconds"]) - rt60) < 1e-8
        )
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one {frontend}@{rt60} condition")
    return selected[0]


def _prepare_plotting():
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
    return plt


def plot_cer(
    summaries: dict[str, dict[str, Any]], *, output: Path
) -> None:
    plt = _prepare_plotting()
    rt60 = (0.2, 0.4, 0.6, 0.8, 1.0)
    x = tuple(range(6))
    figure, axes = plt.subplots(
        1, 2, figsize=(10.2, 4.2), sharey=True, constrained_layout=True
    )
    for axis, frontend, title in (
        (axes[0], "raw", "Reference channel (Raw)"),
        (axes[1], "m_wpe_10", "4-channel M-WPE-10"),
    ):
        for key, (label, color) in MODELS.items():
            summary = summaries[key]
            clean = 100.0 * float(
                _condition(summary, frontend="clean", rt60=None)["cer"]
            )
            values = [clean] + [
                100.0
                * float(_condition(summary, frontend=frontend, rt60=value)["cer"])
                for value in rt60
            ]
            axis.plot(
                x,
                values,
                label=label,
                color=color,
                marker="o",
                linewidth=2.1,
                markersize=5.2,
            )
        axis.set_title(title)
        axis.set_xticks(x, ("Clean", "0.2", "0.4", "0.6", "0.8", "1.0"))
        axis.set_xlabel("Target RT60 (s)")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Normalized corpus CER (%)")
    axes[1].legend(frameon=False, loc="upper left")
    figure.suptitle("Whisper-small on Held-out Simulated Reverberation")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, format="png")
    plt.close(figure)


def plot_errors(
    summaries: dict[str, dict[str, Any]], *, output: Path
) -> None:
    plt = _prepare_plotting()
    labels: list[str] = []
    substitutions: list[float] = []
    deletions: list[float] = []
    insertions: list[float] = []
    for key, (label, _) in MODELS.items():
        for frontend, suffix in (("raw", "Raw"), ("m_wpe_10", "M-WPE")):
            row = _condition(summaries[key], frontend=frontend, rt60=1.0)
            labels.append(f"{label}\n{suffix}")
            substitutions.append(100.0 * float(row["substitution_rate"]))
            deletions.append(100.0 * float(row["deletion_rate"]))
            insertions.append(100.0 * float(row["insertion_rate"]))

    import numpy as np

    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9.2, 4.5), constrained_layout=True)
    axis.bar(x, substitutions, color="#dc2626", label="Substitution")
    axis.bar(x, deletions, bottom=substitutions, color="#d97706", label="Deletion")
    bottom = np.asarray(substitutions) + np.asarray(deletions)
    axis.bar(x, insertions, bottom=bottom, color="#7c3aed", label="Insertion")
    axis.set_title("Error Composition at RT60 = 1.0 s")
    axis.set_ylabel("Error rate (%)")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, format="png")
    plt.close(figure)


def main() -> None:
    args = arguments()
    for output in (args.cer_output, args.error_output):
        if output.suffix.lower() != ".png":
            raise ValueError("documentation figure output must be PNG")
    summaries = {
        "w0": _load(args.w0_summary),
        "clean": _load(args.clean_summary),
        "mct": _load(args.mct_summary),
    }
    plot_cer(summaries, output=args.cer_output)
    plot_errors(summaries, output=args.error_output)


if __name__ == "__main__":
    main()
