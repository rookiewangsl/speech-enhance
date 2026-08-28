"""Matched greedy-versus-beam comparison for frozen Whisper outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from robust_asr.scoring import (
    aggregate_character_errors,
    paired_bootstrap_cer_delta,
    score_characters,
)


def _greedy_key(row: Mapping[str, Any]) -> tuple[str, str, float | None]:
    rt60 = row.get("target_rt60_seconds")
    return (
        str(row["utterance_id"]),
        str(row["frontend"]),
        None if rt60 is None else float(rt60),
    )


def _beam_baseline_key(row: Mapping[str, Any]) -> tuple[str, str, float | None]:
    condition = str(row["condition"])
    if condition == "clean_level":
        return str(row["utterance_id"]), "clean", None
    if condition == "reverb_raw":
        return (
            str(row["utterance_id"]),
            "raw",
            float(row["reverb_rt60_seconds"]),
        )
    if condition == "reverb_m_wpe_10":
        return (
            str(row["utterance_id"]),
            "m_wpe_10",
            float(row["reverb_rt60_seconds"]),
        )
    raise ValueError(f"unsupported beam audit condition: {condition}")


def compare_greedy_with_beam(
    greedy_rows: Sequence[Mapping[str, Any]],
    beam_rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Compare matched normalized hypotheses without rerunning greedy decoding."""

    if not greedy_rows or not beam_rows:
        raise ValueError("both greedy and beam result sets must be non-empty")
    greedy = {_greedy_key(row): row for row in greedy_rows}
    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for beam in beam_rows:
        key = _beam_baseline_key(beam)
        baseline = greedy.get(key)
        if baseline is None:
            raise ValueError(f"no matched greedy result for {key}")
        if str(baseline["reference"]) != str(beam["reference"]):
            raise ValueError(f"reference mismatch for {key}")
        grouped.setdefault(str(beam["condition"]), []).append((baseline, beam))

    comparisons: list[dict[str, Any]] = []
    for condition, pairs in sorted(grouped.items()):
        greedy_scores = {
            str(beam["utterance_id"]): score_characters(
                str(baseline["reference"]), str(baseline["hypothesis"])
            )
            for baseline, beam in pairs
        }
        beam_scores = {
            str(beam["utterance_id"]): score_characters(
                str(beam["reference"]), str(beam["hypothesis"])
            )
            for _, beam in pairs
        }
        greedy_total = aggregate_character_errors(greedy_scores.values())
        beam_total = aggregate_character_errors(beam_scores.values())
        interval = paired_bootstrap_cer_delta(
            greedy_scores,
            beam_scores,
            draws=draws,
            seed=seed,
        )
        comparisons.append(
            {
                "condition": condition,
                "utterances": len(pairs),
                "greedy": greedy_total.as_dict(),
                "beam": beam_total.as_dict(),
                "beam_minus_greedy_cer": interval.as_dict(),
            }
        )
    return {
        "schema_version": 1,
        "purpose": "frozen_decoder_audit",
        "greedy_result_rows": len(greedy_rows),
        "beam_result_rows": len(beam_rows),
        "comparisons": comparisons,
    }
