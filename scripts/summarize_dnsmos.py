"""Summarize paired DNSMOS P.835 scores by condition, noise, SNR, and speaker."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("sig", "bak", "ovrl")
GROUP_FIELDS = ("noise", "snr_db", "speaker_id")


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


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty DNSMOS paired CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def index_paired(
    rows: list[dict[str, Any]], baseline: str
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    protocol_versions: set[str] = set()
    model_hashes: set[str] = set()
    config_digests: set[str] = set()
    for row in rows:
        utterance_id = str(row.get("id", ""))
        condition = str(row.get("condition", ""))
        if not utterance_id or not condition or condition in indexed[utterance_id]:
            raise ValueError(f"invalid or duplicate DNSMOS identity: {(utterance_id, condition)}")
        for metric in METRICS:
            value = float(row[metric])
            if not np.isfinite(value):
                raise ValueError(f"non-finite DNSMOS {metric}: {(utterance_id, condition)}")
        indexed[utterance_id][condition] = row
        protocol_versions.add(str(row.get("protocol_version")))
        model_hashes.add(str(row.get("model_sha256")))
        config_digests.add(str(row.get("config_digest")))
    if len(protocol_versions) != 1 or len(model_hashes) != 1 or len(config_digests) != 1:
        raise ValueError("DNSMOS rows mix protocol, model, or config identities")
    conditions = sorted({condition for group in indexed.values() for condition in group})
    if baseline not in conditions:
        raise ValueError(f"baseline condition is absent: {baseline}")
    expected = set(conditions)
    for utterance_id, group in indexed.items():
        if set(group) != expected:
            raise ValueError(f"incomplete paired conditions for {utterance_id}")
        baseline_row = group[baseline]
        for condition, row in group.items():
            for field in GROUP_FIELDS:
                if row.get(field) != baseline_row.get(field):
                    raise ValueError(
                        f"paired metadata mismatch for {utterance_id}/{condition}: {field}"
                    )
    return conditions, dict(indexed)


def bootstrap_ci(values: np.ndarray, *, draws: int, seed: int) -> list[float]:
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return [float(lower), float(upper)]


def aggregate(
    paired_rows: list[dict[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {"utterances": len(paired_rows)}
    for metric in METRICS:
        scores = np.asarray([float(row[metric]) for row in paired_rows], dtype=np.float64)
        deltas = np.asarray(
            [float(row[f"delta_{metric}_vs_baseline"]) for row in paired_rows],
            dtype=np.float64,
        )
        result[metric] = {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "p10": float(np.quantile(scores, 0.10)),
            "p90": float(np.quantile(scores, 0.90)),
            "mean_delta_vs_baseline": float(np.mean(deltas)),
            "median_delta_vs_baseline": float(np.median(deltas)),
            "mean_delta_vs_baseline_ci95": bootstrap_ci(deltas, draws=draws, seed=seed),
        }
    rtf = np.asarray([float(row["rtf_processing_only"]) for row in paired_rows])
    result["rtf_processing_only"] = {
        "median": float(np.median(rtf)),
        "p95": float(np.quantile(rtf, 0.95)),
        "max": float(np.max(rtf)),
    }
    return result


def summarize(
    rows: list[dict[str, Any]],
    *,
    baseline: str = "noisy",
    draws: int = 2000,
    seed: int = 20260724,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conditions, indexed = index_paired(rows, baseline)
    paired: list[dict[str, Any]] = []
    for utterance_id, group in sorted(indexed.items()):
        baseline_row = group[baseline]
        for condition in conditions:
            row = group[condition]
            paired.append(
                {
                    "id": utterance_id,
                    "speaker_id": row.get("speaker_id", "unknown"),
                    "split": row.get("split", "unknown"),
                    "noise": row.get("noise", "unknown"),
                    "snr_db": row.get("snr_db", "unknown"),
                    "condition": condition,
                    "sig": float(row["sig"]),
                    "bak": float(row["bak"]),
                    "ovrl": float(row["ovrl"]),
                    "delta_sig_vs_baseline": float(row["sig"]) - float(baseline_row["sig"]),
                    "delta_bak_vs_baseline": float(row["bak"]) - float(baseline_row["bak"]),
                    "delta_ovrl_vs_baseline": float(row["ovrl"]) - float(baseline_row["ovrl"]),
                    "num_hops": int(row["num_hops"]),
                    "rtf_processing_only": float(row["rtf_processing_only"]),
                }
            )

    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": rows[0]["protocol_version"],
        "model_sha256": rows[0]["model_sha256"],
        "config_digest": rows[0]["config_digest"],
        "baseline": baseline,
        "conditions": conditions,
        "utterances": len(indexed),
        "rows": len(paired),
        "bootstrap": {
            "method": "paired_utterance_percentile_mean_delta",
            "draws": draws,
            "seed": seed,
            "confidence_level": 0.95,
        },
        "overall": {},
        "by_noise": {},
        "by_snr_db": {},
        "by_speaker_id": {},
    }
    for condition in conditions:
        condition_rows = [row for row in paired if row["condition"] == condition]
        result["overall"][condition] = aggregate(condition_rows, draws=draws, seed=seed)
    section_for = {
        "noise": "by_noise",
        "snr_db": "by_snr_db",
        "speaker_id": "by_speaker_id",
    }
    for field, section_name in section_for.items():
        values = sorted({str(row[field]) for row in paired})
        for value in values:
            result[section_name][value] = {}
            for condition in conditions:
                group = [
                    row
                    for row in paired
                    if row["condition"] == condition and str(row[field]) == value
                ]
                result[section_name][value][condition] = aggregate(
                    group, draws=draws, seed=seed
                )
    return result, paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--baseline", default="noisy")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    args = parser.parse_args()
    report, paired = summarize(
        read_jsonl(args.input),
        baseline=args.baseline,
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    atomic_json(args.output, report)
    atomic_csv(args.paired_output, paired)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
