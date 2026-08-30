#!/usr/bin/env python3
"""Paired contrasts under frozen deterministic number-normalization policies."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from robust_asr.manifest import read_jsonl
from robust_asr.paths import require_data_root
from robust_asr.rescore import deterministic_number_score
from robust_asr.scoring import (
    CharacterErrorCounts,
    aggregate_character_errors,
    paired_bootstrap_cer_delta,
    score_characters,
)
from robust_asr.text import ChineseTextNormalizer

ScorePolicy = Literal["formal", "contextual_cardinal", "digit_by_digit"]
_ROBUST_RT60 = (0.4, 0.6, 0.8, 1.0)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=FILE",
        help="Provide w0, clean_lora, mct_lora and paraformer JSONL files.",
    )
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _basename(value: str, option: str) -> str:
    if Path(value).name != value:
        raise ValueError(f"{option} must be a basename")
    return value


def _model_spec(value: str) -> tuple[str, str]:
    label, separator, filename = value.partition("=")
    if not separator or not label or not filename:
        raise ValueError("--model must use LABEL=FILE")
    return label, _basename(filename, "--model file")


def _raw_text(row: Mapping[str, Any], raw_key: str, fallback_key: str) -> str:
    source = row.get(raw_key)
    if source is None:
        source = row[fallback_key]
    return str(source)


def _score_row(
    row: Mapping[str, Any],
    *,
    policy: ScorePolicy,
    normalizer: ChineseTextNormalizer,
) -> CharacterErrorCounts:
    reference_raw = _raw_text(row, "reference_raw", "reference")
    hypothesis_raw = _raw_text(row, "hypothesis_raw", "hypothesis")
    if policy == "formal":
        return score_characters(
            normalizer.normalize(reference_raw),
            normalizer.normalize(hypothesis_raw),
        )
    return deterministic_number_score(
        reference_raw,
        hypothesis_raw,
        normalizer=normalizer,
        policy=policy,
    )


def condition_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    frontend: str,
    rt60_values: tuple[float, ...] | None,
    policy: ScorePolicy,
    normalizer: ChineseTextNormalizer | None = None,
) -> dict[str, CharacterErrorCounts]:
    """Aggregate one or more RT60 rows per utterance for paired bootstrap."""

    text_normalizer = normalizer or ChineseTextNormalizer()
    grouped: dict[str, list[CharacterErrorCounts]] = {}
    for row in rows:
        if str(row["frontend"]) != frontend:
            continue
        rt60 = row.get("target_rt60_seconds")
        selected = (
            rt60 is None
            if rt60_values is None
            else rt60 is not None
            and any(abs(float(rt60) - value) < 1e-8 for value in rt60_values)
        )
        if not selected:
            continue
        grouped.setdefault(str(row["utterance_id"]), []).append(
            _score_row(row, policy=policy, normalizer=text_normalizer)
        )
    expected = 1 if rt60_values is None else len(rt60_values)
    if not grouped:
        raise ValueError(f"no rows for frontend={frontend}, rt60={rt60_values}")
    incomplete = [key for key, values in grouped.items() if len(values) != expected]
    if incomplete:
        raise ValueError(
            f"{len(incomplete)} utterances do not have {expected} selected rows"
        )
    return {
        key: aggregate_character_errors(values) for key, values in grouped.items()
    }


def _contrast(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_frontend: str,
    candidate_frontend: str,
    rt60_values: tuple[float, ...] | None,
    policy: ScorePolicy,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    normalizer = ChineseTextNormalizer()
    baseline = condition_scores(
        baseline_rows,
        frontend=baseline_frontend,
        rt60_values=rt60_values,
        policy=policy,
        normalizer=normalizer,
    )
    candidate = condition_scores(
        candidate_rows,
        frontend=candidate_frontend,
        rt60_values=rt60_values,
        policy=policy,
        normalizer=normalizer,
    )
    baseline_total = aggregate_character_errors(baseline.values())
    candidate_total = aggregate_character_errors(candidate.values())
    interval = paired_bootstrap_cer_delta(
        baseline,
        candidate,
        draws=draws,
        seed=seed,
    )
    return {
        "utterance_clusters": len(baseline),
        "baseline_cer": baseline_total.cer,
        "candidate_cer": candidate_total.cer,
        "candidate_minus_baseline_cer": candidate_total.cer - baseline_total.cer,
        "paired_bootstrap_95_ci": interval.as_dict(),
    }


def compare_models(
    models: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    draws: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    required = {"w0", "clean_lora", "mct_lora", "paraformer"}
    if set(models) != required:
        raise ValueError(f"model labels must be exactly {sorted(required)}")
    definitions = {
        "w0_wpe_robust": ("w0", "raw", "w0", "m_wpe_10", _ROBUST_RT60),
        "clean_lora_wpe_robust": (
            "clean_lora",
            "raw",
            "clean_lora",
            "m_wpe_10",
            _ROBUST_RT60,
        ),
        "mct_lora_wpe_robust": (
            "mct_lora",
            "raw",
            "mct_lora",
            "m_wpe_10",
            _ROBUST_RT60,
        ),
        "paraformer_wpe_robust": (
            "paraformer",
            "raw",
            "paraformer",
            "m_wpe_10",
            _ROBUST_RT60,
        ),
        "mct_minus_clean_lora_robust_raw": (
            "clean_lora",
            "raw",
            "mct_lora",
            "raw",
            _ROBUST_RT60,
        ),
        "clean_lora_minus_w0_clean": (
            "w0",
            "clean",
            "clean_lora",
            "clean",
            None,
        ),
        "clean_lora_minus_w0_robust_raw": (
            "w0",
            "raw",
            "clean_lora",
            "raw",
            _ROBUST_RT60,
        ),
    }
    comparisons: dict[str, Any] = {}
    for name, definition in definitions.items():
        baseline_model, baseline_frontend, candidate_model, candidate_frontend, rt60 = (
            definition
        )
        comparisons[name] = {
            policy: _contrast(
                models[baseline_model],
                models[candidate_model],
                baseline_frontend=baseline_frontend,
                candidate_frontend=candidate_frontend,
                rt60_values=rt60,
                policy=policy,
                draws=draws,
                seed=seed,
            )
            for policy in ("formal", "contextual_cardinal", "digit_by_digit")
        }
    return {
        "schema_version": 1,
        "purpose": "paired_number_normalization_sensitivity",
        "draws": draws,
        "seed": seed,
        "bootstrap_unit": "utterance_cluster_across_selected_rt60_values",
        "comparisons": comparisons,
    }


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    models: dict[str, list[dict[str, Any]]] = {}
    for spec in args.model:
        label, filename = _model_spec(spec)
        if label in models:
            raise ValueError(f"duplicate model label: {label}")
        models[label] = read_jsonl(root / "outputs" / filename)
    summary = compare_models(models, draws=args.draws, seed=args.seed)
    output_name = _basename(args.output_name, "--output-name")
    destination = root / "outputs" / output_name
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(f"wrote {len(summary['comparisons'])} paired contrasts: {destination}")


if __name__ == "__main__":
    main()
