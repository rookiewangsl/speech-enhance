"""Measure agreement and conflict between paired DNSMOS and ASR outcomes."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import rankdata


METRICS = ("sig", "bak", "ovrl")
GROUP_FIELDS = ("noise", "snr_db")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty object JSONL: {path}")
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _single_identity(rows: list[dict[str, Any]], field: str, label: str) -> str:
    values = {str(row.get(field, "")) for row in rows}
    if len(values) != 1 or "" in values:
        raise ValueError(f"{label} rows mix or omit {field}")
    return next(iter(values))


def _index(
    rows: list[dict[str, Any]], *, label: str
) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        utterance_id = str(row.get("id", ""))
        condition = str(row.get("condition", ""))
        if not utterance_id or not condition or condition in indexed[utterance_id]:
            raise ValueError(f"invalid or duplicate {label} identity: {(utterance_id, condition)}")
        indexed[utterance_id][condition] = row
    return dict(indexed)


def _correlation(left: np.ndarray, right: np.ndarray, *, ranks: bool) -> float | None:
    if ranks:
        left = rankdata(left)
        right = rankdata(right)
    if left.size < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty DNSMOS/ASR group")
    error_deltas = np.asarray([row["delta_errors_vs_noisy"] for row in rows], dtype=np.float64)
    asr_benefit = -error_deltas
    baseline_errors = int(sum(row["noisy_errors"] for row in rows))
    condition_errors = int(sum(row["condition_errors"] for row in rows))
    reference_words = int(sum(row["reference_words"] for row in rows))
    result: dict[str, Any] = {
        "utterances": len(rows),
        "reference_words": reference_words,
        "asr": {
            "noisy_errors": baseline_errors,
            "condition_errors": condition_errors,
            "noisy_corpus_wer": baseline_errors / reference_words,
            "condition_corpus_wer": condition_errors / reference_words,
            "corpus_wer_delta_percentage_points": 100.0
            * (condition_errors - baseline_errors)
            / reference_words,
            "utterances_improved": int(np.sum(error_deltas < 0)),
            "utterances_equal": int(np.sum(error_deltas == 0)),
            "utterances_harmed": int(np.sum(error_deltas > 0)),
        },
        "metrics": {},
    }
    for metric in METRICS:
        deltas = np.asarray([row[f"delta_{metric}_vs_noisy"] for row in rows], dtype=np.float64)
        metric_improved = deltas > 0
        asr_harmed = error_deltas > 0
        conflict = metric_improved & asr_harmed
        metric_improved_count = int(np.sum(metric_improved))
        asr_harmed_count = int(np.sum(asr_harmed))
        result["metrics"][metric] = {
            "mean_delta_vs_noisy": float(np.mean(deltas)),
            "median_delta_vs_noisy": float(np.median(deltas)),
            "utterances_score_improved": metric_improved_count,
            "score_improved_fraction": metric_improved_count / len(rows),
            "score_improved_but_asr_harmed": int(np.sum(conflict)),
            "conflict_fraction_of_all": float(np.mean(conflict)),
            "asr_harmed_fraction_among_score_improved": (
                float(np.sum(conflict)) / metric_improved_count
                if metric_improved_count
                else None
            ),
            "score_improved_fraction_among_asr_harmed": (
                float(np.sum(conflict)) / asr_harmed_count if asr_harmed_count else None
            ),
            "pearson_with_asr_error_reduction": _correlation(deltas, asr_benefit, ranks=False),
            "spearman_with_asr_error_reduction": _correlation(deltas, asr_benefit, ranks=True),
        }
    return result


def analyze(
    dnsmos_rows: list[dict[str, Any]],
    asr_rows: list[dict[str, Any]],
    *,
    conditions: Iterable[str],
    baseline: str = "noisy",
) -> dict[str, Any]:
    conditions = tuple(dict.fromkeys(conditions))
    if not conditions or baseline in conditions:
        raise ValueError("conditions must be non-empty and exclude the baseline")
    protocol = _single_identity(dnsmos_rows, "protocol_version", "DNSMOS")
    dnsmos_model = _single_identity(dnsmos_rows, "model_sha256", "DNSMOS")
    dnsmos_config = _single_identity(dnsmos_rows, "config_digest", "DNSMOS")
    asr_model = _single_identity(asr_rows, "model_sha256", "ASR")
    asr_config = _single_identity(asr_rows, "asr_config_digest", "ASR")
    if any(row.get("status") != "completed" for row in asr_rows):
        raise ValueError("ASR input contains rows that are not completed")
    dnsmos = _index(dnsmos_rows, label="DNSMOS")
    asr = _index(asr_rows, label="ASR")
    if set(dnsmos) != set(asr):
        raise ValueError("DNSMOS and ASR utterance ID sets differ")

    paired: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
    required = {baseline, *conditions}
    for utterance_id in sorted(dnsmos):
        if not required.issubset(dnsmos[utterance_id]) or not required.issubset(asr[utterance_id]):
            raise ValueError(f"missing required condition for {utterance_id}")
        dns_baseline = dnsmos[utterance_id][baseline]
        asr_baseline = asr[utterance_id][baseline]
        reference_words = int(asr_baseline["reference_words"])
        if reference_words <= 0:
            raise ValueError(f"non-positive reference word count for {utterance_id}")
        for condition in conditions:
            dns_row = dnsmos[utterance_id][condition]
            asr_row = asr[utterance_id][condition]
            if int(asr_row["reference_words"]) != reference_words:
                raise ValueError(f"ASR reference word count mismatch for {utterance_id}/{condition}")
            for field in ("noise", "snr_db", "speaker_id"):
                values = {
                    dns_baseline.get(field),
                    dns_row.get(field),
                    asr_baseline.get(field),
                    asr_row.get(field),
                }
                if len(values) != 1:
                    raise ValueError(f"paired metadata mismatch for {utterance_id}/{condition}: {field}")
            entry = {
                "id": utterance_id,
                "condition": condition,
                "noise": dns_row["noise"],
                "snr_db": dns_row["snr_db"],
                "speaker_id": dns_row["speaker_id"],
                "reference_words": reference_words,
                "noisy_errors": int(asr_baseline["errors"]),
                "condition_errors": int(asr_row["errors"]),
            }
            entry["delta_errors_vs_noisy"] = entry["condition_errors"] - entry["noisy_errors"]
            for metric in METRICS:
                baseline_score = float(dns_baseline[metric])
                condition_score = float(dns_row[metric])
                if not np.isfinite(baseline_score) or not np.isfinite(condition_score):
                    raise ValueError(f"non-finite DNSMOS {metric} for {utterance_id}/{condition}")
                entry[f"delta_{metric}_vs_noisy"] = condition_score - baseline_score
            paired[condition].append(entry)

    result: dict[str, Any] = {
        "schema_version": 1,
        "baseline": baseline,
        "conditions": list(conditions),
        "utterances": len(dnsmos),
        "interpretation": {
            "asr_harmed": "condition word errors > paired noisy word errors",
            "dnsmos_improved": "condition score > paired noisy score",
            "conflict": "DNSMOS improved while ASR word errors increased",
            "correlation_target": "ASR error reduction = noisy errors - condition errors",
        },
        "identity": {
            "dnsmos_protocol_version": protocol,
            "dnsmos_model_sha256": dnsmos_model,
            "dnsmos_config_digest": dnsmos_config,
            "asr_model_sha256": asr_model,
            "asr_config_digest": asr_config,
        },
        "overall": {},
        "by_noise": {},
        "by_snr_db": {},
    }
    for condition in conditions:
        condition_rows = paired[condition]
        result["overall"][condition] = _aggregate(condition_rows)
        for field in GROUP_FIELDS:
            section = result[f"by_{field}"]
            for value in sorted({str(row[field]) for row in condition_rows}):
                section.setdefault(value, {})[condition] = _aggregate(
                    [row for row in condition_rows if str(row[field]) == value]
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnsmos", type=Path, required=True)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--baseline", default="noisy")
    args = parser.parse_args()
    report = analyze(
        read_jsonl(args.dnsmos),
        read_jsonl(args.asr),
        conditions=args.conditions,
        baseline=args.baseline,
    )
    atomic_json(args.output, report)
    print(json.dumps(report["overall"], indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
