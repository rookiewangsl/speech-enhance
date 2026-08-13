"""Score ASR hypotheses and summarize corpus WER by experimental condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from speech_frontend.asr.scoring import ErrorCounts, aggregate_error_counts, score_text


GROUP_FIELDS = ("noise", "snr_db", "speaker_id", "split", "reference_length_bin")
FROZEN_CONDITIONS = {"clean", "noisy", "mcra_dd_wiener", "rnnoise_r3"}
DEFAULT_BOOTSTRAP_DRAWS = 2_000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _unique_by_id(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        utterance_id = str(row.get("id", "")).strip()
        if not utterance_id:
            raise ValueError(f"{label} row is missing a non-empty 'id'")
        if utterance_id in indexed:
            raise ValueError(f"duplicate {label} id: {utterance_id}")
        indexed[utterance_id] = row
    return indexed


def merge_references_and_hypotheses(
    references: list[dict[str, Any]], hypotheses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join reference rows to hypotheses and reject incomplete conditions."""

    reference_by_id = _unique_by_id(references, "reference")
    merged: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for hypothesis in hypotheses:
        utterance_id = str(hypothesis.get("id", "")).strip()
        condition = str(hypothesis.get("condition", "")).strip()
        if not utterance_id or not condition:
            raise ValueError("hypothesis row requires non-empty 'id' and 'condition'")
        key = (utterance_id, condition)
        if key in seen_keys:
            raise ValueError(f"duplicate hypothesis key: {key}")
        seen_keys.add(key)
        if utterance_id not in reference_by_id:
            raise ValueError(f"hypothesis has no reference: {utterance_id}")
        status = hypothesis.get("status")
        if status not in (None, "ok", "success", "completed"):
            raise ValueError(
                f"hypothesis {utterance_id}/{condition} has non-success status {status!r}"
            )
        reference = reference_by_id[utterance_id]
        for field in ("reference_raw", "reference_normalized"):
            reference_value = reference.get(field)
            hypothesis_value = hypothesis.get(field)
            if (
                reference_value is not None
                and hypothesis_value is not None
                and reference_value != hypothesis_value
            ):
                raise ValueError(
                    f"hypothesis {utterance_id}/{condition} disagrees with "
                    f"authoritative {field}"
                )
        row = dict(hypothesis)
        # The separate official reference manifest is authoritative for source
        # text and transcript provenance.  ASR-side normalized reference is
        # retained only when the prepare-stage manifest intentionally has raw
        # text alone.
        for field, value in reference.items():
            if (
                field == "id"
                or field.startswith("reference_")
                or field.startswith("transcript_")
            ):
                row[field] = value
        merged.append(row)

    if not merged:
        raise ValueError("no hypotheses to summarize")
    ids_by_condition: dict[str, set[str]] = defaultdict(set)
    for row in merged:
        ids_by_condition[str(row["condition"])].add(str(row["id"]))
    first_condition = sorted(ids_by_condition)[0]
    expected_ids = ids_by_condition[first_condition]
    for condition, actual_ids in sorted(ids_by_condition.items()):
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise ValueError(
                f"condition {condition!r} is not paired with all references; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    return merged


def validate_frozen_experiment(rows: list[dict[str, Any]]) -> None:
    """Enforce the paired four-condition identity required by the main protocol."""

    conditions = {str(row.get("condition", "")) for row in rows}
    if conditions != FROZEN_CONDITIONS:
        raise ValueError(
            f"frozen protocol requires conditions {sorted(FROZEN_CONDITIONS)}, "
            f"got {sorted(conditions)}"
        )
    identity_fields = (
        "model_sha256",
        "asr_config_digest",
        "evaluator_code_sha256",
        "runtime_identity_digest",
        "device",
    )
    for field in identity_fields:
        values = {row.get(field) for row in rows}
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"frozen protocol requires non-empty {field}")
        if len(values) != 1:
            raise ValueError(f"frozen protocol rows disagree in {field}")
    if {row["device"] for row in rows} != {"cpu"}:
        raise ValueError("frozen v1 main results require device=cpu")

    rows_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_id[str(row["id"])].append(row)
        reference_raw = row.get("reference_raw")
        reference_sha = row.get("reference_raw_sha256")
        if not isinstance(reference_raw, str) or not isinstance(reference_sha, str):
            raise ValueError("frozen protocol requires raw reference text and SHA-256")
        if hashlib.sha256(reference_raw.encode("utf-8")).hexdigest() != reference_sha:
            raise ValueError(f"reference_raw_sha256 mismatch for {row['id']}")
    paired_fields = (
        "reference_raw_sha256",
        "reference_raw",
        "reference_normalized",
        "speaker_id",
        "split",
        "noise",
        "snr_db",
    )
    for utterance_id, paired_rows in rows_by_id.items():
        if len(paired_rows) != len(FROZEN_CONDITIONS):
            raise ValueError(f"utterance {utterance_id} is not present in all conditions")
        for field in paired_fields:
            values = {json.dumps(row.get(field), sort_keys=True) for row in paired_rows}
            if len(values) != 1:
                raise ValueError(
                    f"utterance {utterance_id} has inconsistent paired {field}"
                )


def _get_text_pair(row: dict[str, Any]) -> tuple[str, str]:
    has_normalized_reference = row.get("reference_normalized") is not None
    has_normalized_hypothesis = row.get("hypothesis_normalized") is not None
    if has_normalized_reference != has_normalized_hypothesis:
        raise ValueError(
            f"row {row.get('id')!r} must provide normalized text for both "
            "reference and hypothesis, or for neither"
        )
    fields = (
        ("reference_normalized", "hypothesis_normalized")
        if has_normalized_reference
        else ("reference_raw", "hypothesis_raw")
    )
    values: list[str] = []
    for field in fields:
        value = row.get(field)
        if value is None:
            raise ValueError(f"row {row.get('id')!r} is missing {field!r}")
        if not isinstance(value, str):
            raise ValueError(
                f"row {row.get('id')!r} field {field!r} must be a string"
            )
        values.append(value)
    return values[0], values[1]


def scoring_text_mode(rows: list[dict[str, Any]]) -> str:
    """Return one corpus-wide text mode and reject mixed scoring policies."""

    if not rows:
        raise ValueError("no rows to score")
    modes: set[str] = set()
    for row in rows:
        _get_text_pair(row)
        modes.add(
            "normalized"
            if row.get("reference_normalized") is not None
            else "raw"
        )
    if len(modes) != 1:
        raise ValueError("a scoring corpus cannot mix normalized and raw text rows")
    return next(iter(modes))


def _nested_threshold(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for field in path:
        if not isinstance(value, dict) or field not in value:
            return None
        value = value[field]
    return value


def resolve_compression_ratio_threshold(
    rows: list[dict[str, Any]], *, fallback: float
) -> tuple[float, str]:
    """Resolve one corpus-wide anomaly threshold from rows or an explicit fallback."""

    if not math.isfinite(fallback) or fallback <= 0:
        raise ValueError("compression-ratio threshold fallback must be finite and positive")
    generated_rows = [
        row
        for row in rows
        if row.get("compression_ratio_threshold_used") is not None
    ]
    generated_values = {
        float(row["compression_ratio_threshold_used"])
        for row in generated_rows
    }
    if generated_values:
        if len(generated_rows) != len(rows):
            raise ValueError("scored rows have partial compression-ratio threshold")
        if len(generated_values) != 1:
            raise ValueError("scored rows disagree in compression-ratio threshold")
        sources = {
            str(row.get("compression_ratio_threshold_source", "row_provenance"))
            for row in rows
        }
        if len(sources) != 1:
            raise ValueError("scored rows disagree in compression-ratio threshold source")
        threshold = next(iter(generated_values))
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("compression-ratio threshold must be finite and positive")
        return threshold, next(iter(sources))

    paths = (
        ("compression_ratio_threshold",),
        ("thresholds", "compression_ratio_threshold"),
        ("decoding", "compression_ratio_threshold"),
        ("asr_config", "thresholds", "compression_ratio_threshold"),
        ("config", "thresholds", "compression_ratio_threshold"),
    )
    per_row: list[float | None] = []
    for row in rows:
        values = {
            float(value)
            for path in paths
            if (value := _nested_threshold(row, path)) is not None
        }
        if len(values) > 1:
            raise ValueError(
                f"row {row.get('id')!r} has conflicting compression-ratio thresholds"
            )
        per_row.append(next(iter(values)) if values else None)
    provenance_values = {value for value in per_row if value is not None}
    if provenance_values:
        if any(value is None for value in per_row):
            raise ValueError("compression-ratio threshold provenance is partial")
        if len(provenance_values) != 1:
            raise ValueError("rows disagree in compression-ratio threshold provenance")
        threshold = next(iter(provenance_values))
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("compression-ratio threshold must be finite and positive")
        return threshold, "row_config_provenance"
    return float(fallback), "cli_fallback"


def _decode_anomaly(
    row: dict[str, Any], threshold: float
) -> tuple[bool | None, int, float | None]:
    """Return utterance anomaly, anomalous segment count, and maximum ratio."""

    if "segments" not in row:
        return None, 0, None
    segments = row["segments"]
    if not isinstance(segments, list):
        raise ValueError(f"row {row.get('id')!r} segments must be a list")
    ratios: list[float] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(
                f"row {row.get('id')!r} segment {index} must be an object"
            )
        value = segment.get("compression_ratio")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(
                f"row {row.get('id')!r} segment {index} has invalid compression_ratio"
            )
        ratios.append(float(value))
    anomalous_segments = sum(value > threshold for value in ratios)
    return anomalous_segments > 0, anomalous_segments, max(ratios, default=None)


def score_rows(
    rows: list[dict[str, Any]],
    *,
    compression_ratio_threshold: float = DEFAULT_COMPRESSION_RATIO_THRESHOLD,
) -> list[dict[str, Any]]:
    scoring_text_mode(rows)
    threshold, threshold_source = resolve_compression_ratio_threshold(
        rows, fallback=compression_ratio_threshold
    )
    scored: list[dict[str, Any]] = []
    for row in rows:
        reference, hypothesis = _get_text_pair(row)
        counts = score_text(reference, hypothesis)
        output = dict(row)
        output.update(counts.as_dict())
        words = counts.reference_words
        output["reference_length_bin"] = (
            "1-4" if words <= 4 else "5-9" if words <= 9 else "10-19" if words <= 19 else "20+"
        )
        anomaly, anomalous_segments, maximum = _decode_anomaly(row, threshold)
        output["compression_ratio_threshold_used"] = threshold
        output["compression_ratio_threshold_source"] = threshold_source
        output["compression_ratio_anomaly"] = anomaly
        output["compression_ratio_anomaly_segments"] = anomalous_segments
        output["max_segment_compression_ratio"] = maximum
        scored.append(output)
    return scored


def _counts_from_row(row: dict[str, Any]) -> ErrorCounts:
    return ErrorCounts(
        int(row["substitutions"]),
        int(row["deletions"]),
        int(row["insertions"]),
        int(row["reference_words"]),
    )


def _aggregate(
    rows: list[dict[str, Any]], *, compression_ratio_threshold: float
) -> dict[str, Any]:
    result = aggregate_error_counts(_counts_from_row(row) for row in rows).as_dict()
    result["utterances"] = len(rows)
    durations = [
        float(row["duration_seconds"])
        for row in rows
        if row.get("duration_seconds") is not None
    ]
    asr_times = [
        float(row["asr_seconds"])
        for row in rows
        if row.get("asr_seconds") is not None
    ]
    end_to_end_times = [
        float(row["end_to_end_seconds"])
        for row in rows
        if row.get("end_to_end_seconds") is not None
    ]
    for field, values in (
        ("duration_seconds", durations),
        ("asr_seconds", asr_times),
        ("end_to_end_seconds", end_to_end_times),
    ):
        if len(values) not in {0, len(rows)}:
            raise ValueError(f"partial timing field in aggregate: {field}")
    if bool(durations) != bool(asr_times):
        raise ValueError("duration_seconds and asr_seconds must be provided together")
    if durations and any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("duration_seconds must be finite and positive")
    for field, values in (
        ("asr_seconds", asr_times),
        ("end_to_end_seconds", end_to_end_times),
    ):
        if values and any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"{field} must be finite and non-negative")
    if end_to_end_times and not durations:
        raise ValueError("end_to_end_seconds requires duration_seconds")
    if len(durations) == len(rows) and len(asr_times) == len(rows):
        total_duration = sum(durations)
        result["audio_seconds"] = total_duration
        result["asr_seconds"] = sum(asr_times)
        result["asr_rtf"] = sum(asr_times) / total_duration if total_duration > 0 else None
        utterance_asr_rtfs = [
            asr_seconds / duration
            for asr_seconds, duration in zip(asr_times, durations, strict=True)
        ]
        result["utterance_asr_rtf_median"] = _percentile(
            utterance_asr_rtfs, 0.5
        )
        result["utterance_asr_rtf_p95"] = _percentile(
            utterance_asr_rtfs, 0.95
        )
        result["utterance_asr_rtf_max"] = max(utterance_asr_rtfs)
        if len(end_to_end_times) == len(rows):
            result["end_to_end_seconds"] = sum(end_to_end_times)
            result["end_to_end_rtf"] = (
                sum(end_to_end_times) / total_duration
                if total_duration > 0
                else None
            )
    anomaly_values = [
        _decode_anomaly(row, compression_ratio_threshold) for row in rows
    ]
    available = [value for value in anomaly_values if value[0] is not None]
    anomaly_utterances = sum(value[0] is True for value in available)
    result["compression_ratio_anomaly_utterances"] = anomaly_utterances
    result["compression_ratio_anomaly_segments"] = sum(
        value[1] for value in available
    )
    result["compression_ratio_anomaly_metadata_utterances"] = len(available)
    result["compression_ratio_anomaly_fraction"] = (
        anomaly_utterances / len(available) if available else None
    )
    return result


def _add_noisy_comparisons(condition_metrics: dict[str, dict[str, Any]]) -> None:
    noisy = condition_metrics.get("noisy")
    if noisy is None:
        return
    noisy_wer = float(noisy["wer"])
    for metrics in condition_metrics.values():
        wer = float(metrics["wer"])
        metrics["absolute_wer_change_vs_noisy"] = wer - noisy_wer
        metrics["relative_wer_reduction_vs_noisy"] = (
            (noisy_wer - wer) / noisy_wer if noisy_wer > 0 else None
        )


def _percentile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated percentile of non-empty ``values``."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def _paired_bootstrap_wer_difference(
    condition_rows: list[dict[str, Any]],
    noisy_rows: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap corpus-WER(condition) minus corpus-WER(noisy) by utterance."""

    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    condition_by_id = _unique_by_id(condition_rows, "condition score")
    noisy_by_id = _unique_by_id(noisy_rows, "noisy score")
    if condition_by_id.keys() != noisy_by_id.keys():
        raise ValueError("bootstrap conditions must contain the same paired utterance ids")
    utterance_ids = sorted(condition_by_id)
    if not utterance_ids:
        raise ValueError("cannot bootstrap an empty condition")

    pairs: list[tuple[int, int, int]] = []
    for utterance_id in utterance_ids:
        condition = _counts_from_row(condition_by_id[utterance_id])
        noisy = _counts_from_row(noisy_by_id[utterance_id])
        if condition.reference_words != noisy.reference_words:
            raise ValueError(
                f"paired reference length differs for utterance {utterance_id!r}"
            )
        for field in (
            "reference_raw_sha256",
            "reference_raw",
            "reference_normalized",
        ):
            condition_value = condition_by_id[utterance_id].get(field)
            noisy_value = noisy_by_id[utterance_id].get(field)
            if condition_value != noisy_value:
                raise ValueError(
                    f"paired reference differs for utterance {utterance_id!r}: {field}"
                )
        pairs.append((condition.errors, noisy.errors, condition.reference_words))

    generator = random.Random(seed)
    differences: list[float] = []
    sample_size = len(pairs)
    for _ in range(draws):
        condition_errors = noisy_errors = reference_words = 0
        for _ in range(sample_size):
            condition_error, noisy_error, words = pairs[
                generator.randrange(sample_size)
            ]
            condition_errors += condition_error
            noisy_errors += noisy_error
            reference_words += words
        differences.append(
            condition_errors / reference_words - noisy_errors / reference_words
        )
    return {
        "lower": _percentile(differences, 0.025),
        "upper": _percentile(differences, 0.975),
    }


def _add_overall_paired_bootstrap(
    metrics: dict[str, dict[str, Any]],
    rows_by_condition: dict[str, list[dict[str, Any]]],
    *,
    draws: int,
    seed: int,
) -> None:
    noisy_rows = rows_by_condition.get("noisy")
    if noisy_rows is None:
        return
    for condition, rows in sorted(rows_by_condition.items()):
        metrics[condition][
            "paired_bootstrap_absolute_wer_change_vs_noisy_ci95"
        ] = _paired_bootstrap_wer_difference(
            rows, noisy_rows, draws=draws, seed=seed
        )


def summarize(
    scored_rows: list[dict[str, Any]],
    *,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    compression_ratio_threshold: float = DEFAULT_COMPRESSION_RATIO_THRESHOLD,
) -> dict[str, Any]:
    """Build condition-level and stratified corpus aggregates."""

    if not scored_rows:
        raise ValueError("no scored rows")
    resolved_threshold, threshold_source = resolve_compression_ratio_threshold(
        scored_rows, fallback=compression_ratio_threshold
    )
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_condition[str(row["condition"])].append(row)
    overall = {
        condition: _aggregate(
            group, compression_ratio_threshold=resolved_threshold
        )
        for condition, group in sorted(by_condition.items())
    }
    _add_noisy_comparisons(overall)
    _add_overall_paired_bootstrap(
        overall,
        by_condition,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )

    result: dict[str, Any] = {
        "scoring": {
            "text_mode": scoring_text_mode(scored_rows),
            "decode_anomaly": {
                "rule": "any_segment_compression_ratio_greater_than_threshold",
                "compression_ratio_threshold": resolved_threshold,
                "threshold_source": threshold_source,
                "cli_fallback": compression_ratio_threshold,
                "anomalies_are_retained_in_wer": True,
            },
        },
        "bootstrap": {
            "method": "paired_utterance_percentile",
            "statistic": "corpus_wer_condition_minus_noisy",
            "confidence_level": 0.95,
            "draws": bootstrap_draws,
            "seed": bootstrap_seed,
        },
        "overall": overall,
    }
    for field in GROUP_FIELDS:
        strata: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in scored_rows:
            value = row.get(field, "unknown")
            value_key = "unknown" if value is None else str(value)
            strata[value_key][str(row["condition"])].append(row)
        section: dict[str, dict[str, Any]] = {}
        for value, condition_rows in sorted(strata.items()):
            condition_metrics = {
                condition: _aggregate(
                    group, compression_ratio_threshold=resolved_threshold
                )
                for condition, group in sorted(condition_rows.items())
            }
            _add_noisy_comparisons(condition_metrics)
            section[value] = condition_metrics
        result[f"by_{field}"] = section
    return result


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute deterministic corpus WER and stratified ASR summaries."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--input",
        type=Path,
        help="Unified JSONL containing reference and hypothesis fields.",
    )
    inputs.add_argument(
        "--hypotheses", type=Path, help="Hypothesis JSONL from evaluate_asr.py."
    )
    parser.add_argument(
        "--references",
        type=Path,
        help="Reference JSONL; required with --hypotheses.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON path.")
    parser.add_argument(
        "--utterance-output",
        type=Path,
        help="Optional JSONL path for utterance-level S/D/I/N/WER.",
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=DEFAULT_BOOTSTRAP_DRAWS,
        help=f"Paired utterance bootstrap draws (default: {DEFAULT_BOOTSTRAP_DRAWS}).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"Paired bootstrap random seed (default: {DEFAULT_BOOTSTRAP_SEED}).",
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=DEFAULT_COMPRESSION_RATIO_THRESHOLD,
        help=(
            "Fallback segment compression-ratio anomaly threshold when rows lack "
            f"config provenance (default: {DEFAULT_COMPRESSION_RATIO_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--allow-raw",
        action="store_true",
        help=(
            "Allow unnormalized raw/raw scoring for diagnostics only. "
            "The frozen main protocol requires normalized text."
        ),
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help=(
            "Allow diagnostic condition subsets or incomplete runtime identity. "
            "The frozen main protocol requires exactly four paired conditions."
        ),
    )
    arguments = parser.parse_args()
    if arguments.hypotheses is not None and arguments.references is None:
        parser.error("--references is required with --hypotheses")
    if arguments.input is not None and arguments.references is not None:
        parser.error("--references can only be used with --hypotheses")
    if arguments.bootstrap_draws <= 0:
        parser.error("--bootstrap-draws must be positive")
    if (
        not math.isfinite(arguments.compression_ratio_threshold)
        or arguments.compression_ratio_threshold <= 0
    ):
        parser.error("--compression-ratio-threshold must be finite and positive")
    return arguments


def main() -> None:
    arguments = parse_args()
    if arguments.input is not None:
        merged = read_jsonl(arguments.input)
    else:
        references = read_jsonl(arguments.references)
        hypotheses = read_jsonl(arguments.hypotheses)
        merged = merge_references_and_hypotheses(references, hypotheses)
    text_mode = scoring_text_mode(merged)
    if text_mode != "normalized" and not arguments.allow_raw:
        raise ValueError(
            "frozen WER requires reference_normalized and hypothesis_normalized; "
            "--allow-raw is diagnostic-only"
        )
    if not arguments.allow_subset:
        validate_frozen_experiment(merged)
    scored = score_rows(
        merged,
        compression_ratio_threshold=arguments.compression_ratio_threshold,
    )
    summary = summarize(
        scored,
        bootstrap_draws=arguments.bootstrap_draws,
        bootstrap_seed=arguments.bootstrap_seed,
        compression_ratio_threshold=arguments.compression_ratio_threshold,
    )
    _atomic_write_text(
        arguments.output,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    if arguments.utterance_output is not None:
        write_jsonl(arguments.utterance_output, scored)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
