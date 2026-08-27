"""End-to-end frozen Whisper reverberation/WPE baseline runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import numpy as np
import soundfile as sf
from scipy.stats import spearmanr

from .acoustics.rir import convolve_multichannel
from .config import canonical_sha256
from .dereverb.frontend import FrontendName, apply_frontend
from .dereverb.wpe import WPEConfig
from .download import sha256_file
from .manifest import read_jsonl, write_jsonl_atomic
from .scoring import (
    CharacterErrorCounts,
    aggregate_character_errors,
    paired_bootstrap_cer_delta,
    score_characters,
)
from .text import ChineseTextNormalizer


class Transcriber(Protocol):
    model_id: str
    device: str

    def transcribe(self, audio: np.ndarray, *, sample_rate: int = 16_000) -> str: ...


@dataclass(frozen=True)
class BaselineProgress:
    """Small progress event for a resumable baseline run."""

    stage: Literal["start", "progress", "complete"]
    completed: int
    total: int
    resumed: int
    generated: int
    elapsed_seconds: float
    utterance_id: str | None = None
    frontend: str | None = None
    rt60_seconds: float | None = None
    inference_seconds: float | None = None


def _stable_rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()


def select_speaker_balanced_count(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Select a small deterministic smoke subset without one-speaker dominance."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in rows:
        row = dict(source)
        grouped.setdefault(str(row["speaker_id"]), []).append(row)
    if not grouped:
        raise ValueError("manifest is empty")
    for values in grouped.values():
        values.sort(key=lambda row: _stable_rank(seed, str(row["utterance_id"])))
    selected: list[dict[str, Any]] = []
    speakers = sorted(grouped, key=lambda value: _stable_rank(seed, value))
    index = 0
    while len(selected) < limit:
        progress = False
        for speaker in speakers:
            if index < len(grouped[speaker]):
                selected.append(grouped[speaker][index])
                progress = True
                if len(selected) == limit:
                    return selected
        if not progress:
            break
        index += 1
    return selected


def _select_rir(
    rows: Sequence[Mapping[str, Any]],
    *,
    utterance_id: str,
    rt60_seconds: float,
    seed: int,
) -> Mapping[str, Any]:
    candidates = [
        row
        for row in rows
        if abs(float(row["target_rt60_seconds"]) - rt60_seconds) < 1e-8
    ]
    if not candidates:
        raise ValueError(f"RIR bank has no target RT60={rt60_seconds}")
    ordered = sorted(candidates, key=lambda row: str(row["rir_id"]))
    integer = int.from_bytes(
        _stable_rank(seed, f"{utterance_id}\0{rt60_seconds}")[:8], "big"
    )
    return ordered[integer % len(ordered)]


def _read_clean(corpus_root: Path, row: Mapping[str, Any]) -> np.ndarray:
    audio, sample_rate = sf.read(
        corpus_root / str(row["audio_path"]),
        dtype="float32",
        always_2d=False,
    )
    if sample_rate != 16_000 or audio.ndim != 1:
        raise ValueError(f"invalid AISHELL audio: {row['audio_path']}")
    return np.asarray(audio, dtype=np.float32)


def _result_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    rt60 = row.get("target_rt60_seconds")
    return (
        str(row["utterance_id"]),
        str(row["frontend"]),
        "clean" if rt60 is None else f"{float(rt60):.6f}",
    )


def _summarize(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in results:
        rt60 = row.get("target_rt60_seconds")
        key = (
            str(row["frontend"]),
            "clean" if rt60 is None else f"{float(rt60):.1f}",
        )
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (frontend, rt60), rows in sorted(grouped.items()):
        aggregate = aggregate_character_errors(
            score_characters(str(row["reference"]), str(row["hypothesis"]))
            for row in rows
        )
        summary.append(
            {
                "frontend": frontend,
                "target_rt60_seconds": None if rt60 == "clean" else float(rt60),
                "utterances": len(rows),
                **aggregate.as_dict(),
            }
        )
    return summary


def _scores_by_condition(
    results: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, float | None], dict[str, CharacterErrorCounts]]:
    output: dict[tuple[str, float | None], dict[str, CharacterErrorCounts]] = {}
    for row in results:
        key = (
            str(row["frontend"]),
            None
            if row.get("target_rt60_seconds") is None
            else float(row["target_rt60_seconds"]),
        )
        output.setdefault(key, {})[str(row["utterance_id"])] = score_characters(
            str(row["reference"]), str(row["hypothesis"])
        )
    return output


def _paired_deltas(
    results: Sequence[Mapping[str, Any]],
    *,
    rt60_seconds: Sequence[float],
    frontends: Sequence[FrontendName],
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    scores = _scores_by_condition(results)
    clean = scores.get(("clean", None))
    comparisons: list[dict[str, Any]] = []
    for rt60 in rt60_seconds:
        raw = scores.get(("raw", float(rt60)))
        pairs: list[tuple[str, dict[str, CharacterErrorCounts] | None, str]] = []
        if clean is not None:
            pairs.append(("clean", clean, "raw"))
        for frontend in frontends:
            if frontend == "raw":
                continue
            pairs.append(("raw", raw, frontend))
        for baseline_name, baseline_scores, candidate_name in pairs:
            candidate_scores = scores.get((candidate_name, float(rt60)))
            if baseline_scores is None or candidate_scores is None:
                continue
            interval = paired_bootstrap_cer_delta(
                baseline_scores,
                candidate_scores,
                draws=draws,
                seed=seed,
            )
            comparisons.append(
                {
                    "target_rt60_seconds": float(rt60),
                    "baseline": baseline_name,
                    "candidate": candidate_name,
                    "candidate_minus_baseline_cer": interval.as_dict(),
                }
            )
    return comparisons


def _first_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return None
        value = value[0]
    number = float(value)
    return number if math.isfinite(number) else None


def _spearman_summary(
    rows: Sequence[Mapping[str, Any]], *, field: str, outcome: str = "cer"
) -> dict[str, float | int | None]:
    pairs = [
        (_first_float(row.get(field)), float(row[outcome]))
        for row in rows
    ]
    valid = [
        (x, y)
        for x, y in pairs
        if x is not None and math.isfinite(y)
    ]
    if len(valid) < 3:
        return {"utterances": len(valid), "spearman_rho": None, "pvalue": None}
    x = np.asarray([pair[0] for pair in valid], dtype=np.float64)
    y = np.asarray([pair[1] for pair in valid], dtype=np.float64)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"utterances": len(valid), "spearman_rho": None, "pvalue": None}
    result = spearmanr(x, y)
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "utterances": len(valid),
        "spearman_rho": rho if math.isfinite(rho) else None,
        "pvalue": pvalue if math.isfinite(pvalue) else None,
    }


def _raw_robustness_analysis(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    clean_cer = {
        str(row["utterance_id"]): float(row["cer"])
        for row in results
        if row.get("frontend") == "clean"
    }
    raw = [
        {
            **row,
            "cer_degradation_from_clean": float(row["cer"])
            - clean_cer[str(row["utterance_id"])],
        }
        for row in results
        if row.get("frontend") == "raw"
        and row.get("target_rt60_seconds") is not None
        and str(row["utterance_id"]) in clean_cer
    ]
    by_rt60: dict[str, Any] = {}
    for rt60 in sorted({float(row["target_rt60_seconds"]) for row in raw}):
        group = [
            row for row in raw if float(row["target_rt60_seconds"]) == rt60
        ]
        drr_values = [
            value
            for row in group
            if (value := _first_float(row.get("reference_drr_db", row.get("drr_db"))))
            is not None
        ]
        enriched_group = [
            {
                **row,
                "analysis_drr_db": row.get(
                    "reference_drr_db", row.get("drr_db")
                ),
            }
            for row in group
        ]
        rir_groups: dict[str, list[Mapping[str, Any]]] = {}
        for row in enriched_group:
            rir_groups.setdefault(str(row["rir_id"]), []).append(row)
        by_rir = [
            {
                "analysis_drr_db": values[0]["analysis_drr_db"],
                "mean_cer_degradation_from_clean": float(
                    np.mean(
                        [
                            float(value["cer_degradation_from_clean"])
                            for value in values
                        ]
                    )
                ),
            }
            for values in rir_groups.values()
        ]
        by_rt60[f"{rt60:.1f}"] = {
            "utterances": len(group),
            "unique_rirs": len(rir_groups),
            "mean_reference_drr_db": float(np.mean(drr_values))
            if drr_values
            else None,
            "drr_cer_spearman": _spearman_summary(
                enriched_group,
                field="analysis_drr_db",
            ),
            "drr_cer_degradation_spearman": _spearman_summary(
                enriched_group,
                field="analysis_drr_db",
                outcome="cer_degradation_from_clean",
            ),
            "drr_mean_degradation_by_rir_spearman": _spearman_summary(
                by_rir,
                field="analysis_drr_db",
                outcome="mean_cer_degradation_from_clean",
            ),
        }
    enriched = [
        {
            **row,
            "analysis_target_rt60": row.get("target_rt60_seconds"),
            "analysis_measured_rt60": row.get(
                "reference_measured_rt60_seconds",
                row.get("measured_rt60_seconds"),
            ),
            "analysis_drr_db": row.get("reference_drr_db", row.get("drr_db")),
        }
        for row in raw
    ]
    return {
        "utterances": len(raw),
        "target_rt60_cer_spearman": _spearman_summary(
            enriched, field="analysis_target_rt60"
        ),
        "target_rt60_degradation_spearman": _spearman_summary(
            enriched,
            field="analysis_target_rt60",
            outcome="cer_degradation_from_clean",
        ),
        "measured_rt60_cer_spearman": _spearman_summary(
            enriched, field="analysis_measured_rt60"
        ),
        "measured_rt60_degradation_spearman": _spearman_summary(
            enriched,
            field="analysis_measured_rt60",
            outcome="cer_degradation_from_clean",
        ),
        "drr_cer_spearman_uncontrolled": _spearman_summary(
            enriched, field="analysis_drr_db"
        ),
        "by_target_rt60": by_rt60,
    }


def run_frozen_baseline(
    *,
    manifest_path: str | Path,
    corpus_root: str | Path,
    rir_manifest_path: str | Path,
    rir_root: str | Path,
    output_path: str | Path,
    transcriber: Transcriber,
    limit: int,
    frontends: Sequence[FrontendName] = (
        "raw",
        "s_wpe_10",
        "s_wpe_40",
        "m_wpe_10",
    ),
    rt60_seconds: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    seed: int = 2026,
    normalizer: ChineseTextNormalizer | None = None,
    bootstrap_draws: int = 10_000,
    checkpoint_every_results: int = 20,
    progress_callback: Callable[[BaselineProgress], None] | None = None,
) -> dict[str, Any]:
    """Run clean plus factorial reverb/frontend inference with atomic resume."""

    if checkpoint_every_results <= 0:
        raise ValueError("checkpoint_every_results must be positive")
    if not frontends or len(set(frontends)) != len(frontends):
        raise ValueError("frontends must be non-empty and unique")
    manifest_rows = read_jsonl(manifest_path)
    selected = select_speaker_balanced_count(manifest_rows, limit=limit, seed=seed)
    rir_rows = read_jsonl(rir_manifest_path)
    destination = Path(output_path)
    existing = read_jsonl(destination) if destination.is_file() else []
    run_identity = canonical_sha256(
        {
            "schema_version": 2,
            "model_id": transcriber.model_id,
            "model_revision": getattr(transcriber, "model_revision", None),
            "utterances": selected,
            "rir_manifest": rir_rows,
            "wpe_backend": "nara_wpe",
            "wpe_config": asdict(WPEConfig()),
            "level_protocol": {
                "target_rms_dbfs": -25.0,
                "peak_headroom_db": 1.0,
            },
            "rt60_seconds": list(map(float, rt60_seconds)),
            "seed": seed,
        }
    )
    incompatible = [
        row
        for row in existing
        if row.get("run_protocol_sha256") != run_identity
    ]
    if incompatible:
        raise ValueError(
            "existing baseline output belongs to a different run protocol; "
            "choose a new output path"
        )
    result_by_key = {_result_key(row): row for row in existing}
    text_normalizer = normalizer or ChineseTextNormalizer()
    corpus = Path(corpus_root)
    rir_directory = Path(rir_root)
    validated_rir_paths: set[Path] = set()
    required_keys = {
        (str(row["utterance_id"]), "clean", "clean") for row in selected
    }
    required_keys.update(
        (
            str(row["utterance_id"]),
            str(frontend),
            f"{float(rt60):.6f}",
        )
        for row in selected
        for rt60 in rt60_seconds
        for frontend in frontends
    )
    resumed = len(required_keys.intersection(result_by_key))
    total = len(required_keys)
    completed = resumed
    generated = 0
    started_run = time.perf_counter()

    def notify(
        stage: Literal["start", "progress", "complete"],
        *,
        row: Mapping[str, Any] | None = None,
        frontend: str | None = None,
        rt60: float | None = None,
        inference_seconds: float | None = None,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            BaselineProgress(
                stage=stage,
                completed=completed,
                total=total,
                resumed=resumed,
                generated=generated,
                elapsed_seconds=time.perf_counter() - started_run,
                utterance_id=None if row is None else str(row["utterance_id"]),
                frontend=frontend,
                rt60_seconds=rt60,
                inference_seconds=inference_seconds,
            )
        )

    notify("start")

    def result_key(
        row: Mapping[str, Any], frontend: str, rt60: float | None
    ) -> tuple[str, str, str]:
        return (
            str(row["utterance_id"]),
            frontend,
            "clean" if rt60 is None else f"{rt60:.6f}",
        )

    def load_full_rir(rir: Mapping[str, Any]) -> np.ndarray:
        path = rir_directory / str(rir["path"])
        if path not in validated_rir_paths:
            expected_sha = rir.get("file_sha256")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise ValueError(f"RIR manifest has no valid file SHA: {path}")
            observed_sha = sha256_file(path)
            if observed_sha != expected_sha:
                raise ValueError(
                    f"RIR file SHA mismatch for {path}: {observed_sha}"
                )
            validated_rir_paths.add(path)
        with np.load(path) as archive:
            full = np.asarray(archive["full"], dtype=np.float64)
        expected_shape = tuple(rir.get("full_shape", ()))
        if full.ndim != 2 or full.shape[0] != 4:
            raise ValueError(f"invalid 4-channel RIR shape: {full.shape}")
        if expected_shape and full.shape != expected_shape:
            raise ValueError(
                f"RIR shape mismatch for {path}: {full.shape} != {expected_shape}"
            )
        return full

    def infer(
        row: Mapping[str, Any],
        waveform: np.ndarray,
        *,
        frontend: str,
        rt60: float | None,
        rir: Mapping[str, Any] | None,
    ) -> None:
        nonlocal completed, generated
        key = result_key(row, frontend, rt60)
        if key in result_by_key:
            return
        started = time.perf_counter()
        hypothesis_raw = transcriber.transcribe(waveform, sample_rate=16_000)
        elapsed = time.perf_counter() - started
        reference = text_normalizer.normalize(str(row["transcript"]))
        hypothesis = text_normalizer.normalize(hypothesis_raw)
        score = score_characters(reference, hypothesis)
        peak_abs = float(np.max(np.abs(waveform), initial=0.0))
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        result = {
            "utterance_id": row["utterance_id"],
            "speaker_id": row["speaker_id"],
            "model_id": transcriber.model_id,
            "model_revision": getattr(transcriber, "model_revision", None),
            "device": transcriber.device,
            "run_protocol_sha256": run_identity,
            "frontend": frontend,
            "target_rt60_seconds": rt60,
            "rir_id": None if rir is None else rir["rir_id"],
            "room_id": None if rir is None else rir["room_id"],
            "source_array_distance_m": None
            if rir is None
            else rir["scene"]["source_array_distance_m"],
            "measured_rt60_seconds": None
            if rir is None
            else rir["measured_rt60_seconds"],
            "reference_measured_rt60_seconds": None
            if rir is None
            else _first_float(rir["measured_rt60_seconds"]),
            "drr_db": None if rir is None else rir["drr_db"],
            "reference_drr_db": None
            if rir is None
            else _first_float(rir["drr_db"]),
            "reference_raw": row.get("transcript_raw", row["transcript"]),
            "reference": reference,
            "hypothesis_raw": hypothesis_raw,
            "hypothesis": hypothesis,
            "audio_samples": int(waveform.size),
            "audio_peak_abs": peak_abs,
            "audio_rms_dbfs": 20.0
            * np.log10(max(rms, np.finfo(float).tiny)),
            "inference_seconds": elapsed,
            **score.as_dict(),
        }
        result_by_key[key] = result
        completed += 1
        generated += 1
        if generated % checkpoint_every_results == 0:
            write_jsonl_atomic(destination, result_by_key.values())
        notify(
            "progress",
            row=row,
            frontend=frontend,
            rt60=rt60,
            inference_seconds=elapsed,
        )

    for row in selected:
        clean = _read_clean(corpus, row)
        clean_normalized = convolve_multichannel(
            clean,
            np.asarray([[1.0]], dtype=np.float64),
        ).signals[0]
        infer(row, clean_normalized, frontend="clean", rt60=None, rir=None)
        for rt60 in rt60_seconds:
            rir = _select_rir(
                rir_rows,
                utterance_id=str(row["utterance_id"]),
                rt60_seconds=float(rt60),
                seed=seed,
            )
            missing_frontends = [
                frontend
                for frontend in frontends
                if result_key(row, frontend, float(rt60)) not in result_by_key
            ]
            if not missing_frontends:
                continue
            full_rir = load_full_rir(rir)
            multichannel = convolve_multichannel(clean, full_rir).signals
            for frontend in missing_frontends:
                waveform = apply_frontend(
                    multichannel,
                    frontend,
                    backend="nara_wpe",
                )
                infer(
                    row,
                    waveform,
                    frontend=frontend,
                    rt60=float(rt60),
                    rir=rir,
                )

    write_jsonl_atomic(destination, result_by_key.values())
    requested_results = [
        row for key, row in result_by_key.items() if key in required_keys
    ]
    summary = {
        "schema_version": 1,
        "model_id": transcriber.model_id,
        "model_revision": getattr(transcriber, "model_revision", None),
        "device": transcriber.device,
        "run_protocol_sha256": run_identity,
        "utterance_limit": limit,
        "frontends": list(frontends),
        "rt60_seconds": list(map(float, rt60_seconds)),
        "result_rows": len(requested_results),
        "resumed_rows": resumed,
        "generated_rows": generated,
        "conditions": _summarize(requested_results),
        "paired_deltas": _paired_deltas(
            requested_results,
            rt60_seconds=rt60_seconds,
            frontends=frontends,
            draws=bootstrap_draws,
            seed=seed,
        ),
        "raw_robustness": _raw_robustness_analysis(requested_results),
    }
    summary_path = destination.with_suffix(".summary.json")
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    notify("complete")
    return summary
