"""Compare frozen v1 ASR with robust v2 recovery and routing decisions."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from speech_frontend.asr.scoring import ErrorCounts, aggregate_error_counts, score_text


CONDITIONS = ("clean", "noisy", "mcra_dd_wiener", "rnnoise_r3")
DEFAULT_BOOTSTRAP_DRAWS = 2_000
DEFAULT_BOOTSTRAP_SEED = 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _index(rows: list[dict[str, Any]], label: str) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("id", "")), str(row.get("condition", "")))
        if not key[0] or key[1] not in CONDITIONS:
            raise ValueError(f"invalid {label} key: {key}")
        if key in indexed:
            raise ValueError(f"duplicate {label} key: {key}")
        indexed[key] = row
    return indexed


def validate_pairing(v1: list[dict[str, Any]], v2: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    first = _index(v1, "v1")
    robust = _index(v2, "v2")
    if set(first) != set(robust):
        raise ValueError("v1 and v2 do not contain the same utterance/condition pairs")
    ids_by_condition: dict[str, set[str]] = defaultdict(set)
    for utterance_id, condition in first:
        ids_by_condition[condition].add(utterance_id)
    if set(ids_by_condition) != set(CONDITIONS) or len({frozenset(ids) for ids in ids_by_condition.values()}) != 1:
        raise ValueError("v1/v2 comparison requires a four-way paired corpus")
    for key in first:
        v1_row, v2_row = first[key], robust[key]
        if v2_row.get("protocol_version") != "asr_whisper_small_en_robust_v2":
            raise ValueError(f"unexpected v2 protocol: {key}")
        if v1_row.get("reference_raw_sha256") != v2_row.get("reference_raw_sha256"):
            raise ValueError(f"reference identity mismatch: {key}")
        first_attempt = v2_row.get("first_pass_attempt")
        if not isinstance(first_attempt, dict):
            raise ValueError(f"v2 is missing first_pass_attempt: {key}")
        for field in ("hypothesis_raw", "hypothesis_normalized", "audio_sha256"):
            if first_attempt.get(field) != v1_row.get(field):
                raise ValueError(f"v2 first pass does not reproduce v1 {field}: {key}")
    for utterance_id in ids_by_condition["noisy"]:
        normalized_references = {
            first[(utterance_id, condition)].get("reference_normalized") for condition in CONDITIONS
        }
        if len(normalized_references) != 1:
            raise ValueError(f"paired normalized reference mismatch: {utterance_id}")
    return first, robust


def _score(reference: str, hypothesis: str) -> ErrorCounts:
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("normalized reference must be non-empty")
    if not isinstance(hypothesis, str):
        raise ValueError("normalized hypothesis must be a string")
    return score_text(reference, hypothesis)


def _score_dict(score: ErrorCounts) -> dict[str, int | float]:
    return score.as_dict()


def _first_passing_retry(row: dict[str, Any]) -> dict[str, Any] | None:
    for attempt in row.get("retry_attempts", []):
        if isinstance(attempt, dict) and attempt.get("diagnostic", {}).get("anomalous") is False:
            return attempt
    return None


def score_comparison(v1_rows: list[dict[str, Any]], v2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first, robust = validate_pairing(v1_rows, v2_rows)
    output: list[dict[str, Any]] = []
    for key in sorted(first):
        v1_row, v2_row = first[key], robust[key]
        reference = str(v1_row["reference_normalized"])
        first_score = _score(reference, str(v1_row["hypothesis_normalized"]))
        noisy_row = first[(key[0], "noisy")]
        noisy_score = _score(reference, str(noisy_row["hypothesis_normalized"]))
        retry = _first_passing_retry(v2_row)
        retry_text = str(retry["hypothesis_normalized"]) if retry is not None else str(v1_row["hypothesis_normalized"])
        retry_score = _score(reference, retry_text)
        accepted = v2_row.get("final_status") == "accepted"
        final_score = _score(reference, str(v2_row["hypothesis_normalized"])) if accepted else None
        output.append(
            {
                "id": key[0],
                "condition": key[1],
                "speaker_id": v1_row.get("speaker_id"),
                "split": v1_row.get("split"),
                "noise": v1_row.get("noise"),
                "snr_db": v1_row.get("snr_db"),
                "reference_normalized": reference,
                "reference_words": first_score.reference_words,
                "first_pass": _score_dict(first_score),
                "paired_v1_noisy": _score_dict(noisy_score),
                "oracle_v1_condition_or_noisy": _score_dict(
                    first_score if first_score.errors <= noisy_score.errors else noisy_score
                ),
                "retry_only": _score_dict(retry_score),
                "final": None if final_score is None else _score_dict(final_score),
                "triggered": bool(v2_row.get("first_pass_anomalous")),
                "trigger_reasons": list(v2_row.get("trigger_reasons", [])),
                "retry_attempts": len(v2_row.get("retry_attempts", [])),
                "retry_accepted": retry is not None,
                "final_status": v2_row.get("final_status"),
                "final_source": v2_row.get("final_source"),
                "first_pass_asr_seconds": float(v2_row["first_pass_asr_seconds"]),
                "recovery_asr_seconds": float(v2_row["recovery_asr_seconds"]),
                "total_service_asr_seconds": float(v2_row["total_service_asr_seconds"]),
                "duration_seconds": float(v2_row["duration_seconds"]),
            }
        )
    return output


def _aggregate(rows: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    scores = []
    accepted_rows = []
    for row in rows:
        value = row.get(score_field)
        if value is None:
            continue
        scores.append(
            ErrorCounts(
                int(value["substitutions"]),
                int(value["deletions"]),
                int(value["insertions"]),
                int(value["reference_words"]),
            )
        )
        accepted_rows.append(row)
    total_reference = sum(int(row["reference_words"]) for row in rows)
    result: dict[str, Any] = {
        "utterances": len(rows),
        "accepted_utterances": len(accepted_rows),
        "coverage": len(accepted_rows) / len(rows),
        "total_reference_words": total_reference,
        "accepted_reference_words": sum(score.reference_words for score in scores),
    }
    if scores:
        counts = aggregate_error_counts(scores)
        result.update(counts.as_dict())
        result["selective_wer"] = counts.wer
        result["full_coverage_wer"] = counts.wer if len(scores) == len(rows) else None
    else:
        result.update({"substitutions": 0, "deletions": 0, "insertions": 0, "errors": 0, "selective_wer": None, "full_coverage_wer": None})
    return result


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _latency(rows: list[dict[str, Any]]) -> dict[str, float]:
    first_seconds = sum(float(row["first_pass_asr_seconds"]) for row in rows)
    final_seconds = sum(float(row["total_service_asr_seconds"]) for row in rows)
    audio_seconds = sum(float(row["duration_seconds"]) for row in rows)
    first_rtfs = [float(row["first_pass_asr_seconds"]) / float(row["duration_seconds"]) for row in rows]
    final_rtfs = [float(row["total_service_asr_seconds"]) / float(row["duration_seconds"]) for row in rows]
    return {
        "audio_seconds": audio_seconds,
        "v1_asr_seconds": first_seconds,
        "v1_corpus_rtf": first_seconds / audio_seconds,
        "v2_service_asr_seconds": final_seconds,
        "v2_service_corpus_rtf": final_seconds / audio_seconds,
        "v2_extra_asr_seconds": final_seconds - first_seconds,
        "v2_service_rtf_median": _percentile(final_rtfs, 0.5),
        "v2_service_rtf_p95": _percentile(final_rtfs, 0.95),
        "v2_service_rtf_max": max(final_rtfs),
        "v1_rtf_median": _percentile(first_rtfs, 0.5),
        "v1_rtf_p95": _percentile(first_rtfs, 0.95),
        "v1_rtf_max": max(first_rtfs),
    }


def _paired_bootstrap_difference(
    rows: list[dict[str, Any]], *, minuend: str, subtrahend: str, draws: int, seed: int
) -> dict[str, float] | None:
    if any(row.get(minuend) is None or row.get(subtrahend) is None for row in rows):
        return None
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(draws):
        sampled = [rows[rng.randrange(len(rows))] for _ in rows]
        minuend_errors = sum(int(row[minuend]["errors"]) for row in sampled)
        subtrahend_errors = sum(int(row[subtrahend]["errors"]) for row in sampled)
        reference_words = sum(int(row["reference_words"]) for row in sampled)
        differences.append((minuend_errors - subtrahend_errors) / reference_words)
    return {"lower": _percentile(differences, 0.025), "upper": _percentile(differences, 0.975)}


def _detector_quality(rows: list[dict[str, Any]], noisy_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        harmful = int(row["first_pass"]["errors"]) > int(noisy_by_id[row["id"]]["first_pass"]["errors"])
        triggered = bool(row["triggered"])
        if harmful and triggered:
            tp += 1
        elif harmful:
            fn += 1
        elif triggered:
            fp += 1
        else:
            tn += 1
    return {
        "label_definition": "v1_condition_errors_greater_than_paired_v1_noisy_errors; evaluation_only",
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def summarize(scored_rows: list[dict[str, Any]], *, bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS, bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict[str, Any]:
    if bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be positive")
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_condition[str(row["condition"])].append(row)
    if set(by_condition) != set(CONDITIONS):
        raise ValueError("robust summary requires four conditions")
    noisy_by_id = {str(row["id"]): row for row in by_condition["noisy"]}
    output: dict[str, Any] = {
        "protocol_version": "asr_whisper_small_en_robust_v2",
        "bootstrap": {
            "draws": bootstrap_draws,
            "seed": bootstrap_seed,
            "statistics": ["corpus_wer_v2_final_minus_v1", "corpus_wer_v2_final_minus_v1_noisy"],
        },
        "conditions": {},
    }
    for condition in CONDITIONS:
        rows = sorted(by_condition[condition], key=lambda row: str(row["id"]))
        source_counts = Counter(str(row["final_source"]) for row in rows)
        reason_counts = Counter(reason for row in rows for reason in row["trigger_reasons"])
        first = _aggregate(rows, "first_pass")
        noisy = _aggregate(rows, "paired_v1_noisy")
        oracle = _aggregate(rows, "oracle_v1_condition_or_noisy")
        retry = _aggregate(rows, "retry_only")
        final = _aggregate(rows, "final")
        if final["full_coverage_wer"] is not None:
            final["absolute_wer_change_vs_v1"] = final["full_coverage_wer"] - first["full_coverage_wer"]
            final["relative_wer_reduction_vs_v1"] = (
                (first["full_coverage_wer"] - final["full_coverage_wer"]) / first["full_coverage_wer"]
                if first["full_coverage_wer"]
                else None
            )
        else:
            final["absolute_wer_change_vs_v1"] = None
            final["relative_wer_reduction_vs_v1"] = None
        if final["full_coverage_wer"] is not None:
            final["absolute_wer_change_vs_v1_noisy"] = final["full_coverage_wer"] - noisy["full_coverage_wer"]
            final["relative_wer_reduction_vs_v1_noisy"] = (
                (noisy["full_coverage_wer"] - final["full_coverage_wer"]) / noisy["full_coverage_wer"]
                if noisy["full_coverage_wer"]
                else None
            )
        else:
            final["absolute_wer_change_vs_v1_noisy"] = None
            final["relative_wer_reduction_vs_v1_noisy"] = None
        condition_summary: dict[str, Any] = {
            "v1_first_pass": first,
            "paired_v1_noisy_baseline": noisy,
            "oracle_v1_condition_or_noisy": {
                **oracle,
                "deployable": False,
                "definition": "per-utterance minimum reference-scored errors between v1 condition and paired v1 noisy",
            },
            "v2_detector_only": first,
            "v2_retry_only": retry,
            "v2_full_policy": final,
            "triggered_utterances": sum(bool(row["triggered"]) for row in rows),
            "trigger_rate": sum(bool(row["triggered"]) for row in rows) / len(rows),
            "trigger_reason_counts": dict(sorted(reason_counts.items())),
            "retry_accepted_utterances": sum(bool(row["retry_accepted"]) for row in rows),
            "total_retry_attempts": sum(int(row["retry_attempts"]) for row in rows),
            "final_source_counts": dict(sorted(source_counts.items())),
            "abstentions": sum(row["final_status"] == "abstained" for row in rows),
            "latency": _latency(rows),
            "paired_bootstrap_final_minus_v1_ci95": _paired_bootstrap_difference(
                rows, minuend="final", subtrahend="first_pass", draws=bootstrap_draws, seed=bootstrap_seed
            ),
            "paired_bootstrap_final_minus_v1_noisy_ci95": _paired_bootstrap_difference(
                rows, minuend="final", subtrahend="paired_v1_noisy", draws=bootstrap_draws, seed=bootstrap_seed
            ),
        }
        if condition in {"mcra_dd_wiener", "rnnoise_r3"}:
            condition_summary["detector_offline_evaluation"] = _detector_quality(rows, noisy_by_id)
        output["conditions"][condition] = condition_summary
    return output


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--utterance-output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    scored = score_comparison(read_jsonl(args.v1), read_jsonl(args.v2))
    summary = summarize(scored, bootstrap_draws=args.bootstrap_draws, bootstrap_seed=args.bootstrap_seed)
    _atomic_write(args.output, json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    _atomic_write(args.utterance_output, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scored))
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
