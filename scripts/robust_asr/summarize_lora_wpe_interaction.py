#!/usr/bin/env python3
"""Summarize the W0/Clean/MCT × Raw/M-WPE paired dev interaction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from robust_asr.experiments import wpe_lora_interaction
from robust_asr.manifest import read_jsonl
from robust_asr.paths import require_data_root
from robust_asr.scoring import (
    CharacterErrorCounts,
    aggregate_character_errors,
    paired_bootstrap_cer_delta,
    paired_bootstrap_cer_interaction,
)


ROBUST_RT60 = (0.4, 0.6, 0.8, 1.0)
HEAVY_RT60 = (0.8, 1.0)
MODEL_FILES = {
    "w0_pretrained": "w0_whisper_dev_model_dev_1000utt_raw_mwpe_v1.jsonl",
    "w1_clean_lora": "clean_lora_whisper_dev_model_dev_1000utt_raw_mwpe_v1.jsonl",
    "w2_mct_lora": "mct_lora_whisper_dev_model_dev_1000utt_raw_mwpe_v1.jsonl",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--output-name", default="lora_wpe_interaction_dev_1000utt_v1.json"
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _score(row: dict[str, Any]) -> CharacterErrorCounts:
    return CharacterErrorCounts(
        substitutions=int(row["substitutions"]),
        deletions=int(row["deletions"]),
        insertions=int(row["insertions"]),
        reference_characters=int(row["reference_characters"]),
    )


def _condition_scores(
    rows: list[dict[str, Any]],
    *,
    frontend: str,
    rt60_seconds: tuple[float, ...],
) -> dict[str, CharacterErrorCounts]:
    selected = [
        row
        for row in rows
        if str(row.get("frontend")) == frontend
        and row.get("target_rt60_seconds") is not None
        and float(row["target_rt60_seconds"]) in rt60_seconds
    ]
    utterances = {str(row["utterance_id"]) for row in selected}
    output = {
        utterance: aggregate_character_errors(
            _score(row)
            for row in selected
            if str(row["utterance_id"]) == utterance
        )
        for utterance in utterances
    }
    expected = len(utterances) * len(rt60_seconds)
    if len(selected) != expected:
        raise ValueError(
            f"{frontend} has {len(selected)} rows, expected {expected} paired rows"
        )
    return output


def _cer(scores: dict[str, CharacterErrorCounts]) -> float:
    return aggregate_character_errors(scores.values()).cer


def _clean_cer(rows: list[dict[str, Any]]) -> float:
    clean = [
        _score(row)
        for row in rows
        if row.get("frontend") == "clean"
        and row.get("target_rt60_seconds") is None
    ]
    return aggregate_character_errors(clean).cer


def _model_summary(
    rows: list[dict[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    per_rt60: dict[str, Any] = {}
    for rt60 in (0.2, 0.4, 0.6, 0.8, 1.0):
        raw = _condition_scores(rows, frontend="raw", rt60_seconds=(rt60,))
        enhanced = _condition_scores(
            rows, frontend="m_wpe_10", rt60_seconds=(rt60,)
        )
        per_rt60[str(rt60)] = {
            "raw_cer": _cer(raw),
            "m_wpe_10_cer": _cer(enhanced),
            "m_wpe_minus_raw_cer": _cer(enhanced) - _cer(raw),
            "m_wpe_minus_raw_bootstrap": paired_bootstrap_cer_delta(
                raw, enhanced, draws=draws, seed=seed
            ).as_dict(),
        }
    groups: dict[str, Any] = {}
    for name, grid in (("robust", ROBUST_RT60), ("heavy", HEAVY_RT60)):
        raw = _condition_scores(rows, frontend="raw", rt60_seconds=grid)
        enhanced = _condition_scores(rows, frontend="m_wpe_10", rt60_seconds=grid)
        groups[name] = {
            "rt60_seconds": list(grid),
            "raw_cer": _cer(raw),
            "m_wpe_10_cer": _cer(enhanced),
            "m_wpe_minus_raw_cer": _cer(enhanced) - _cer(raw),
            "m_wpe_minus_raw_bootstrap": paired_bootstrap_cer_delta(
                raw, enhanced, draws=draws, seed=seed
            ).as_dict(),
        }
    return {
        "model_revision": sorted({str(row["model_revision"]) for row in rows}),
        "clean_cer": _clean_cer(rows),
        "per_rt60": per_rt60,
        **groups,
    }


def build_summary(
    model_rows: dict[str, list[dict[str, Any]]],
    *,
    draws: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    if set(model_rows) != set(MODEL_FILES):
        raise ValueError("summary requires W0, Clean-LoRA, and MCT-LoRA")
    models = {
        name: _model_summary(rows, draws=draws, seed=seed)
        for name, rows in model_rows.items()
    }
    interactions: dict[str, Any] = {}
    for candidate in ("w1_clean_lora", "w2_mct_lora"):
        baseline_rows = model_rows["w0_pretrained"]
        candidate_rows = model_rows[candidate]
        baseline_raw = _condition_scores(
            baseline_rows, frontend="raw", rt60_seconds=ROBUST_RT60
        )
        baseline_wpe = _condition_scores(
            baseline_rows, frontend="m_wpe_10", rt60_seconds=ROBUST_RT60
        )
        candidate_raw = _condition_scores(
            candidate_rows, frontend="raw", rt60_seconds=ROBUST_RT60
        )
        candidate_wpe = _condition_scores(
            candidate_rows, frontend="m_wpe_10", rt60_seconds=ROBUST_RT60
        )
        interactions[candidate] = {
            "definition": "delta_wpe(candidate)-delta_wpe(w0)",
            "value": wpe_lora_interaction(
                pretrained_raw_cer=_cer(baseline_raw),
                pretrained_m_wpe_cer=_cer(baseline_wpe),
                mct_raw_cer=_cer(candidate_raw),
                mct_m_wpe_cer=_cer(candidate_wpe),
            ),
            "bootstrap": paired_bootstrap_cer_interaction(
                baseline_raw,
                baseline_wpe,
                candidate_raw,
                candidate_wpe,
                draws=draws,
                seed=seed,
            ).as_dict(),
        }
    clean_rows = model_rows["w1_clean_lora"]
    mct_rows = model_rows["w2_mct_lora"]
    model_deltas: dict[str, Any] = {}
    for frontend in ("raw", "m_wpe_10"):
        clean = _condition_scores(
            clean_rows, frontend=frontend, rt60_seconds=ROBUST_RT60
        )
        mct = _condition_scores(
            mct_rows, frontend=frontend, rt60_seconds=ROBUST_RT60
        )
        model_deltas[frontend] = {
            "mct_minus_clean_cer": _cer(mct) - _cer(clean),
            "bootstrap": paired_bootstrap_cer_delta(
                clean, mct, draws=draws, seed=seed
            ).as_dict(),
        }
    return {
        "schema_version": 1,
        "utterances": len(
            _condition_scores(
                model_rows["w0_pretrained"],
                frontend="raw",
                rt60_seconds=ROBUST_RT60,
            )
        ),
        "bootstrap_draws": draws,
        "seed": seed,
        "models": models,
        "wpe_model_interactions": interactions,
        "mct_minus_clean": model_deltas,
        "test_split_accessed": False,
    }


def main() -> None:
    args = arguments()
    if Path(args.output_name).name != args.output_name:
        raise ValueError("--output-name must be a basename")
    root = require_data_root(args.data_root)
    model_rows = {
        name: read_jsonl(root / "outputs" / filename)
        for name, filename in MODEL_FILES.items()
    }
    summary = build_summary(
        model_rows,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    output_path = root / "outputs" / args.output_name
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
