"""Parallel, resumable direct-target metrics for frozen WPE conditions."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

import numpy as np

from robust_asr.acoustics.rir import convolve_multichannel
from robust_asr.baseline import (
    _read_clean,
    _select_rir,
    select_speaker_balanced_count,
)
from robust_asr.config import canonical_sha256
from robust_asr.dereverb.frontend import FrontendName, apply_frontend
from robust_asr.dereverb.wpe import WPEConfig
from robust_asr.manifest import read_jsonl, write_jsonl_atomic
from robust_asr.signal_metrics import scale_invariant_sdr, stoi_score


@dataclass(frozen=True)
class FrontendMetricProgress:
    completed_jobs: int
    total_jobs: int
    generated_rows: int
    elapsed_seconds: float


def _load_rir(root: Path, row: Mapping[str, Any], key: str) -> np.ndarray:
    path = (root / str(row["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"RIR path escapes bank root: {path}") from exc
    with np.load(path) as archive:
        values = np.asarray(archive[key], dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 4:
        raise ValueError(f"invalid {key} RIR shape: {path} {values.shape}")
    if tuple(row.get(f"{key}_shape", ())) != values.shape:
        raise ValueError(f"{key} RIR shape disagrees with manifest: {path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} RIR contains NaN or infinity: {path}")
    return values


def _metric_job(
    corpus_root: str,
    rir_root: str,
    utterance: Mapping[str, Any],
    rir: Mapping[str, Any],
    rt60: float,
    frontends: Sequence[FrontendName],
) -> list[dict[str, Any]]:
    clean = _read_clean(Path(corpus_root), utterance)
    bank_root = Path(rir_root)
    full = _load_rir(bank_root, rir, "full")
    direct = _load_rir(bank_root, rir, "direct")
    multichannel = convolve_multichannel(clean, full).signals
    direct_target = convolve_multichannel(clean, direct).signals[0]
    rows: list[dict[str, Any]] = []
    for frontend in frontends:
        estimate = apply_frontend(
            multichannel,
            frontend,
            backend="nara_wpe",
        )
        rows.append(
            {
                "utterance_id": str(utterance["utterance_id"]),
                "speaker_id": str(utterance["speaker_id"]),
                "frontend": frontend,
                "target_rt60_seconds": rt60,
                "rir_id": str(rir["rir_id"]),
                "room_id": str(rir["room_id"]),
                "reference_drr_db": float(rir["drr_db"][0]),
                "source_array_distance_m": float(
                    rir["scene"]["source_array_distance_m"]
                ),
                "si_sdr_db": scale_invariant_sdr(direct_target, estimate),
                "stoi": stoi_score(direct_target, estimate),
            }
        )
    return rows


def _paired_mean_interval(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    if baseline.keys() != candidate.keys() or not baseline:
        raise ValueError("paired metric rows must have identical non-empty keys")
    identifiers = sorted(baseline)
    differences = np.asarray(
        [candidate[key] - baseline[key] for key in identifiers],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    chunk = 1_000
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        indices = rng.integers(0, differences.size, size=(count, differences.size))
        samples[start : start + count] = differences[indices].mean(axis=1)
    lower, median, upper = np.quantile(samples, (0.025, 0.5, 0.975))
    return {
        "draws": draws,
        "seed": seed,
        "lower": float(lower),
        "median": float(median),
        "upper": float(upper),
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["frontend"]), float(row["target_rt60_seconds"])), []
        ).append(row)
    conditions: list[dict[str, Any]] = []
    for (frontend, rt60), values in sorted(groups.items()):
        condition: dict[str, Any] = {
            "frontend": frontend,
            "target_rt60_seconds": rt60,
            "utterances": len(values),
        }
        for metric in ("si_sdr_db", "stoi"):
            metric_values = np.asarray(
                [float(row[metric]) for row in values], dtype=np.float64
            )
            condition[metric] = {
                "mean": float(np.mean(metric_values)),
                "median": float(np.median(metric_values)),
                "q25": float(np.quantile(metric_values, 0.25)),
                "q75": float(np.quantile(metric_values, 0.75)),
            }
        conditions.append(condition)

    paired: list[dict[str, Any]] = []
    rt60_values = sorted({float(row["target_rt60_seconds"]) for row in rows})
    frontend_values = sorted({str(row["frontend"]) for row in rows})
    for rt60 in rt60_values:
        raw_rows = {
            str(row["utterance_id"]): row
            for row in rows
            if row["frontend"] == "raw"
            and float(row["target_rt60_seconds"]) == rt60
        }
        for frontend in frontend_values:
            if frontend == "raw":
                continue
            candidate_rows = {
                str(row["utterance_id"]): row
                for row in rows
                if row["frontend"] == frontend
                and float(row["target_rt60_seconds"]) == rt60
            }
            paired.append(
                {
                    "target_rt60_seconds": rt60,
                    "baseline": "raw",
                    "candidate": frontend,
                    "candidate_minus_raw": {
                        metric: _paired_mean_interval(
                            {
                                key: float(value[metric])
                                for key, value in raw_rows.items()
                            },
                            {
                                key: float(value[metric])
                                for key, value in candidate_rows.items()
                            },
                            draws=draws,
                            seed=seed,
                        )
                        for metric in ("si_sdr_db", "stoi")
                    },
                }
            )
    return {"conditions": conditions, "paired_deltas": paired}


def run_frontend_signal_metrics(
    *,
    manifest_path: str | Path,
    corpus_root: str | Path,
    rir_manifest_path: str | Path,
    rir_root: str | Path,
    rir_validation_path: str | Path,
    output_path: str | Path,
    limit: int = 500,
    frontends: Sequence[FrontendName] = (
        "raw",
        "s_wpe_10",
        "s_wpe_40",
        "m_wpe_10",
    ),
    rt60_seconds: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    workers: int = 32,
    seed: int = 2026,
    bootstrap_draws: int = 10_000,
    checkpoint_every_jobs: int = 20,
    progress_callback: Callable[[FrontendMetricProgress], None] | None = None,
) -> dict[str, Any]:
    """Compute WPE metrics relative to the matched direct-only RIR target."""

    if limit <= 0 or workers <= 0 or bootstrap_draws <= 0:
        raise ValueError("limit, workers, and bootstrap draws must be positive")
    if checkpoint_every_jobs <= 0:
        raise ValueError("checkpoint interval must be positive")
    if not frontends or len(set(frontends)) != len(frontends):
        raise ValueError("frontends must be non-empty and unique")
    validation = json.loads(Path(rir_validation_path).read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or not validation.get("verify_files"):
        raise ValueError("signal metrics require passed full-file RIR validation")
    manifest_rows = read_jsonl(manifest_path)
    selected = select_speaker_balanced_count(manifest_rows, limit=limit, seed=seed)
    rir_rows = read_jsonl(rir_manifest_path)
    validation_dev_sha = validation["splits"]["dev"]["manifest_sha256"]
    if validation_dev_sha != canonical_sha256(rir_rows):
        raise ValueError("RIR validation does not match the selected dev manifest")
    jobs = [
        (
            row,
            _select_rir(
                rir_rows,
                utterance_id=str(row["utterance_id"]),
                rt60_seconds=float(rt60),
                seed=seed,
            ),
            float(rt60),
        )
        for row in selected
        for rt60 in rt60_seconds
    ]
    run_identity = canonical_sha256(
        {
            "schema_version": 1,
            "utterances": selected,
            "rir_manifest_sha256": validation_dev_sha,
            "frontends": list(frontends),
            "rt60_seconds": list(map(float, rt60_seconds)),
            "target": "matched_direct_only_reference_channel_0",
            "wpe_backend": "nara_wpe",
            "wpe_config": asdict(WPEConfig()),
            "metrics": ["si_sdr_db", "stoi"],
            "seed": seed,
        }
    )
    destination = Path(output_path)
    existing = read_jsonl(destination) if destination.is_file() else []
    if any(row.get("run_protocol_sha256") != run_identity for row in existing):
        raise ValueError("existing signal metrics belong to another run protocol")
    result_by_key = {
        (
            str(row["utterance_id"]),
            float(row["target_rt60_seconds"]),
            str(row["frontend"]),
        ): row
        for row in existing
    }
    pending = [
        (row, rir, rt60)
        for row, rir, rt60 in jobs
        if any(
            (str(row["utterance_id"]), rt60, str(frontend)) not in result_by_key
            for frontend in frontends
        )
    ]
    started = time.perf_counter()
    completed_jobs = len(jobs) - len(pending)
    generated_rows = 0

    def notify() -> None:
        if progress_callback is not None:
            progress_callback(
                FrontendMetricProgress(
                    completed_jobs=completed_jobs,
                    total_jobs=len(jobs),
                    generated_rows=generated_rows,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )

    notify()
    if pending:
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
        ) as executor:
            future_jobs = {
                executor.submit(
                    _metric_job,
                    str(Path(corpus_root)),
                    str(Path(rir_root)),
                    row,
                    rir,
                    rt60,
                    tuple(frontends),
                ): (str(row["utterance_id"]), rt60)
                for row, rir, rt60 in pending
            }
            for future in as_completed(future_jobs):
                rows = future.result()
                for row in rows:
                    enriched = {**row, "run_protocol_sha256": run_identity}
                    key = (
                        str(enriched["utterance_id"]),
                        float(enriched["target_rt60_seconds"]),
                        str(enriched["frontend"]),
                    )
                    result_by_key[key] = enriched
                    generated_rows += 1
                completed_jobs += 1
                if completed_jobs % checkpoint_every_jobs == 0:
                    write_jsonl_atomic(destination, result_by_key.values())
                notify()
    write_jsonl_atomic(destination, result_by_key.values())
    required = {
        (str(row["utterance_id"]), rt60, str(frontend))
        for row, _, rt60 in jobs
        for frontend in frontends
    }
    requested_rows = [row for key, row in result_by_key.items() if key in required]
    if len(requested_rows) != len(required):
        raise RuntimeError("signal metric run finished with missing result rows")
    aggregate = _summarize(
        requested_rows,
        draws=bootstrap_draws,
        seed=seed,
    )
    summary = {
        "schema_version": 1,
        "run_protocol_sha256": run_identity,
        "utterance_limit": limit,
        "result_rows": len(requested_rows),
        "workers": workers,
        "target": "matched_direct_only_reference_channel_0",
        "frontends": list(frontends),
        "rt60_seconds": list(map(float, rt60_seconds)),
        **aggregate,
    }
    summary_path = destination.with_suffix(".summary.json")
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    return summary
