#!/usr/bin/env python3
"""Plot held-out Robust CER effect sizes and paired confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robust_asr.manifest import read_jsonl
from robust_asr.scoring import (
    CharacterErrorCounts,
    aggregate_character_errors,
    paired_bootstrap_cer_delta,
)


ROBUST_RT60 = (0.4, 0.6, 0.8, 1.0)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper-interaction", type=Path, required=True)
    parser.add_argument("--paraformer-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _score(row: dict[str, Any]) -> CharacterErrorCounts:
    return CharacterErrorCounts(
        substitutions=int(row["substitutions"]),
        deletions=int(row["deletions"]),
        insertions=int(row["insertions"]),
        reference_characters=int(row["reference_characters"]),
    )


def _condition_scores(
    rows: list[dict[str, Any]], frontend: str
) -> dict[str, CharacterErrorCounts]:
    selected = [
        row
        for row in rows
        if row.get("frontend") == frontend
        and row.get("target_rt60_seconds") is not None
        and float(row["target_rt60_seconds"]) in ROBUST_RT60
    ]
    utterances = sorted({str(row["utterance_id"]) for row in selected})
    output = {
        utterance: aggregate_character_errors(
            _score(row)
            for row in selected
            if str(row["utterance_id"]) == utterance
        )
        for utterance in utterances
    }
    if len(selected) != len(utterances) * len(ROBUST_RT60):
        raise ValueError(f"incomplete {frontend} Robust matrix")
    return output


def _effect(
    label: str, value: float, interval: dict[str, Any], category: str
) -> dict[str, Any]:
    return {
        "label": label,
        "value": 100.0 * value,
        "lower": 100.0 * float(interval["lower"]),
        "upper": 100.0 * float(interval["upper"]),
        "category": category,
    }


def main() -> None:
    args = arguments()
    if args.output.suffix.lower() != ".png":
        raise ValueError("documentation figure output must be PNG")
    whisper = json.loads(args.whisper_interaction.read_text(encoding="utf-8"))
    effects: list[dict[str, Any]] = []
    model_labels = (
        ("w0_pretrained", "M-WPE on W0"),
        ("w1_clean_lora", "M-WPE on Clean-LoRA"),
        ("w2_mct_lora", "M-WPE on MCT-LoRA"),
    )
    for key, label in model_labels:
        robust = whisper["models"][key]["robust"]
        effects.append(
            _effect(
                label,
                float(robust["m_wpe_minus_raw_cer"]),
                robust["m_wpe_minus_raw_bootstrap"],
                "frontend",
            )
        )

    paraformer_rows = read_jsonl(args.paraformer_results)
    para_raw = _condition_scores(paraformer_rows, "raw")
    para_wpe = _condition_scores(paraformer_rows, "m_wpe_10")
    para_interval = paired_bootstrap_cer_delta(
        para_raw, para_wpe, draws=10_000, seed=2026
    )
    para_value = (
        aggregate_character_errors(para_wpe.values()).cer
        - aggregate_character_errors(para_raw.values()).cer
    )
    effects.append(
        _effect(
            "M-WPE on Paraformer",
            para_value,
            para_interval.as_dict(),
            "frontend",
        )
    )
    for frontend, label in (
        ("raw", "MCT vs Clean on Raw"),
        ("m_wpe_10", "MCT vs Clean after M-WPE"),
    ):
        item = whisper["mct_minus_clean"][frontend]
        effects.append(
            _effect(
                label,
                float(item["mct_minus_clean_cer"]),
                item["bootstrap"],
                "training",
            )
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"frontend": "#047857", "training": "#d97706"}
    markers = {"frontend": "o", "training": "s"}
    y = np.arange(len(effects))[::-1]
    figure, axis = plt.subplots(figsize=(8.9, 4.9), constrained_layout=True)
    for position, item in zip(y, effects, strict=True):
        value = float(item["value"])
        lower = float(item["lower"])
        upper = float(item["upper"])
        axis.errorbar(
            value,
            position,
            xerr=[[value - lower], [upper - value]],
            fmt=markers[str(item["category"])],
            color=colors[str(item["category"])],
            markersize=6.5,
            capsize=3.0,
            linewidth=1.8,
        )
        axis.annotate(
            f"{value:+.2f} pp",
            (upper, position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.7,
        )
    axis.axvline(0.0, color="#111827", linewidth=1.2, linestyle="--")
    axis.set_yticks(y, [str(item["label"]) for item in effects])
    axis.set_xlabel("Candidate minus baseline CER (percentage points)")
    axis.set_title("Held-out Robust CER Effects with 95% Paired Bootstrap Intervals")
    axis.grid(axis="x", alpha=0.22)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.text(
        0.01,
        0.02,
        "Negative values indicate improvement; an interval crossing zero is not significant.",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#4b5563",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, format="png")
    plt.close(figure)


if __name__ == "__main__":
    main()
