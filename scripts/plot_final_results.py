#!/usr/bin/env python3
"""Plot the frozen dev/holdout RNNoise controller comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "r3_official": "R3 official",
    "r4a_fixed_050": "R4 conservative",
    "r4d_correction_-9db_vad030": "R4 aggressive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-summary", required=True, type=Path)
    parser.add_argument("--holdout-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    args = parse_args()
    dev = read_summary(args.dev_summary)
    holdout = read_summary(args.holdout_summary)
    labels = list(METHODS.values())
    x = np.arange(len(labels))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for offset, split_name, summary, color in (
        (-width / 2, "Dev", dev, "#4C78A8"),
        (width / 2, "Holdout", holdout, "#F58518"),
    ):
        si_sdri = [summary[key]["mean_si_sdri_db"] for key in METHODS]
        stoi = [summary[key]["mean_stoi_improvement"] for key in METHODS]
        axes[0].bar(x + offset, si_sdri, width, label=split_name, color=color)
        axes[1].bar(x + offset, stoi, width, label=split_name, color=color)

    axes[0].set_title("Mean SI-SDR improvement")
    axes[0].set_ylabel("dB")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Mean STOI change")
    axes[1].axhline(0.0, color="black", linewidth=0.8)

    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)

    fig.suptitle("Frozen RNNoise controller comparison")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
