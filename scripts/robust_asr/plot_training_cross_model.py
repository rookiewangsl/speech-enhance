#!/usr/bin/env python3
"""Plot LoRA, training-data, and Paraformer ablations as a PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-eval", type=Path, required=True)
    parser.add_argument("--encoder-decoder-eval", type=Path, required=True)
    parser.add_argument("--clean-eval", type=Path, required=True)
    parser.add_argument("--mct-eval", type=Path, required=True)
    parser.add_argument("--paraformer-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty evaluation file: {path}")
    return rows


def _best_eval(path: Path) -> dict[str, Any]:
    return min(_read_jsonl(path), key=lambda row: float(row["reverb_cer"]))


def _metrics(row: dict[str, Any]) -> np.ndarray:
    return 100.0 * np.asarray(
        [row["clean_cer"], row["reverb_cer"], row["heavy_cer"]],
        dtype=np.float64,
    )


def _grouped_bars(
    axis: Any,
    *,
    first: np.ndarray,
    second: np.ndarray,
    first_label: str,
    second_label: str,
    first_color: str,
    second_color: str,
    title: str,
) -> None:
    positions = np.arange(3)
    width = 0.36
    bars_first = axis.bar(
        positions - width / 2,
        first,
        width,
        label=first_label,
        color=first_color,
        hatch="//",
    )
    bars_second = axis.bar(
        positions + width / 2,
        second,
        width,
        label=second_label,
        color=second_color,
    )
    axis.set_xticks(positions, ("Clean", "Reverb", "Heavy"))
    axis.set_ylabel("Normalized corpus CER (%)")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, fontsize=8.5)
    axis.bar_label(bars_first, fmt="%.2f", padding=2, fontsize=8)
    axis.bar_label(bars_second, fmt="%.2f", padding=2, fontsize=8)
    top = float(max(np.max(first), np.max(second)))
    axis.set_ylim(0.0, top * 1.22)


def _paraformer_values(path: Path) -> tuple[float, list[float], list[float]]:
    payload = json.loads(path.read_text())
    clean = None
    values: dict[tuple[str, float], float] = {}
    for row in payload["conditions"]:
        frontend = str(row["frontend"])
        rt60 = row["target_rt60_seconds"]
        if frontend == "clean":
            clean = 100.0 * float(row["cer"])
        elif frontend in {"raw", "m_wpe_10"} and rt60 is not None:
            values[(frontend, float(rt60))] = 100.0 * float(row["cer"])
    if clean is None:
        raise ValueError("Paraformer summary has no clean condition")
    rt60_values = (0.2, 0.4, 0.6, 0.8, 1.0)
    raw = [values[("raw", rt60)] for rt60 in rt60_values]
    m_wpe = [values[("m_wpe_10", rt60)] for rt60 in rt60_values]
    return clean, raw, m_wpe


def main() -> None:
    args = arguments()
    if args.output.suffix.lower() != ".png":
        raise ValueError("documentation figure output must be PNG")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    encoder = _metrics(_best_eval(args.encoder_eval))
    encoder_decoder = _metrics(_best_eval(args.encoder_decoder_eval))
    clean_lora = _metrics(_best_eval(args.clean_eval))
    mct_lora = _metrics(_best_eval(args.mct_eval))
    para_clean, para_raw, para_m_wpe = _paraformer_values(args.paraformer_summary)

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    _grouped_bars(
        axes[0],
        first=encoder,
        second=encoder_decoder,
        first_label="Encoder Q/V",
        second_label="Encoder + decoder Q/V",
        first_color="#6b7280",
        second_color="#7c3aed",
        title="(a) 5 h / 500-step LoRA placement",
    )
    _grouped_bars(
        axes[1],
        first=clean_lora,
        second=mct_lora,
        first_label="Clean LoRA",
        second_label="MCT LoRA",
        first_color="#2563eb",
        second_color="#047857",
        title="(b) 20 h formal dev evaluation",
    )

    rt60_values = np.asarray((0.2, 0.4, 0.6, 0.8, 1.0))
    axes[2].plot(
        rt60_values,
        para_raw,
        color="#6b7280",
        marker="o",
        linewidth=2.0,
        label="Raw",
    )
    axes[2].plot(
        rt60_values,
        para_m_wpe,
        color="#047857",
        marker="D",
        linewidth=2.0,
        label="M-WPE-10",
    )
    axes[2].axhline(
        para_clean,
        color="#111827",
        linestyle=":",
        linewidth=1.4,
        label=f"Clean ({para_clean:.2f}%)",
    )
    axes[2].set_xticks(rt60_values)
    axes[2].set_xlabel("Target RT60 (s)")
    axes[2].set_ylabel("Normalized corpus CER (%)")
    axes[2].set_title("(c) Frozen Paraformer cross-model test")
    axes[2].grid(alpha=0.22)
    axes[2].legend(frameon=False, fontsize=8.5)

    figure.suptitle("Training Ablations and Cross-Model Validation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, format="png")
    plt.close(figure)


if __name__ == "__main__":
    main()
