"""Resumable input, dereverberation, and decoding audits for frozen Whisper."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from robust_asr.acoustics.rir import convolve_multichannel
from robust_asr.baseline import (
    BaselineProgress,
    _read_clean,
    _select_rir,
    select_speaker_balanced_count,
)
from robust_asr.config import canonical_sha256
from robust_asr.dereverb.frontend import apply_frontend
from robust_asr.dereverb.wpe import WPEConfig
from robust_asr.download import sha256_file
from robust_asr.manifest import read_jsonl, write_jsonl_atomic
from robust_asr.scoring import (
    CharacterErrorCounts,
    aggregate_character_errors,
    paired_bootstrap_cer_delta,
    score_characters,
)
from robust_asr.text import ChineseTextNormalizer

AuditCondition = Literal[
    "clean_original",
    "clean_level",
    "direct_raw",
    "direct_s_wpe_10",
    "direct_s_wpe_40",
    "direct_m_wpe_10",
    "reverb_raw",
    "reverb_m_wpe_10",
]

AUDIT_CONDITIONS: tuple[AuditCondition, ...] = (
    "clean_original",
    "clean_level",
    "direct_raw",
    "direct_s_wpe_10",
    "direct_s_wpe_40",
    "direct_m_wpe_10",
    "reverb_raw",
    "reverb_m_wpe_10",
)


class AuditTranscriber(Protocol):
    model_id: str
    model_revision: str
    device: str
    num_beams: int

    def transcribe(self, audio: np.ndarray, *, sample_rate: int = 16_000) -> str: ...


def _condition_frontend(condition: AuditCondition) -> str:
    return {
        "direct_raw": "raw",
        "direct_s_wpe_10": "s_wpe_10",
        "direct_s_wpe_40": "s_wpe_40",
        "direct_m_wpe_10": "m_wpe_10",
        "reverb_raw": "raw",
        "reverb_m_wpe_10": "m_wpe_10",
    }.get(condition, "clean")


def _load_rir_array(
    rir_root: Path,
    row: Mapping[str, Any],
    key: Literal["full", "direct"],
    validated_paths: set[Path],
) -> np.ndarray:
    path = (rir_root / str(row["path"])).resolve()
    try:
        path.relative_to(rir_root.resolve())
    except ValueError as exc:
        raise ValueError(f"RIR path escapes configured root: {path}") from exc
    if path not in validated_paths:
        expected = row.get("file_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"RIR manifest has no valid SHA-256: {path}")
        if sha256_file(path) != expected:
            raise ValueError(f"RIR SHA-256 mismatch: {path}")
        validated_paths.add(path)
    with np.load(path) as archive:
        values = np.asarray(archive[key], dtype=np.float64)
    expected_shape = tuple(row.get(f"{key}_shape", ()))
    if values.ndim != 2 or values.shape[0] != 4:
        raise ValueError(f"invalid {key} RIR shape: {values.shape}")
    if expected_shape and values.shape != expected_shape:
        raise ValueError(
            f"{key} RIR shape mismatch: {values.shape} != {expected_shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} RIR contains NaN or infinite values")
    return values


def _summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    return [
        {
            "condition": condition,
            "utterances": len(values),
            **aggregate_character_errors(
                score_characters(str(row["reference"]), str(row["hypothesis"]))
                for row in values
            ).as_dict(),
        }
        for condition, values in sorted(grouped.items())
    ]


def _paired_comparisons(
    rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> list[dict[str, Any]]:
    scores: dict[str, dict[str, CharacterErrorCounts]] = {}
    for row in rows:
        scores.setdefault(str(row["condition"]), {})[str(row["utterance_id"])] = (
            score_characters(str(row["reference"]), str(row["hypothesis"]))
        )
    requested = (
        ("clean_original", "clean_level"),
        ("clean_level", "direct_raw"),
        ("direct_raw", "direct_s_wpe_10"),
        ("direct_raw", "direct_s_wpe_40"),
        ("direct_raw", "direct_m_wpe_10"),
        ("reverb_raw", "reverb_m_wpe_10"),
    )
    output: list[dict[str, Any]] = []
    for baseline, candidate in requested:
        if baseline not in scores or candidate not in scores:
            continue
        interval = paired_bootstrap_cer_delta(
            scores[baseline],
            scores[candidate],
            draws=draws,
            seed=seed,
        )
        output.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "candidate_minus_baseline_cer": interval.as_dict(),
            }
        )
    return output


def run_whisper_input_audit(
    *,
    manifest_path: str | Path,
    corpus_root: str | Path,
    rir_manifest_path: str | Path,
    rir_root: str | Path,
    output_path: str | Path,
    transcriber: AuditTranscriber,
    limit: int,
    conditions: Sequence[AuditCondition],
    direct_rir_rt60_seconds: float = 0.2,
    reverb_rt60_seconds: float = 0.8,
    seed: int = 2026,
    bootstrap_draws: int = 10_000,
    checkpoint_every_results: int = 20,
    progress_callback: Callable[[BaselineProgress], None] | None = None,
) -> dict[str, Any]:
    """Run matched clean/direct/reverb conditions without changing W0 results."""

    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("audit conditions must be non-empty and unique")
    if any(condition not in AUDIT_CONDITIONS for condition in conditions):
        raise ValueError("unknown audit condition")
    if checkpoint_every_results <= 0:
        raise ValueError("checkpoint interval must be positive")
    manifest_rows = read_jsonl(manifest_path)
    selected = select_speaker_balanced_count(manifest_rows, limit=limit, seed=seed)
    rir_rows = read_jsonl(rir_manifest_path)
    run_identity = canonical_sha256(
        {
            "schema_version": 1,
            "model_id": transcriber.model_id,
            "model_revision": transcriber.model_revision,
            "num_beams": transcriber.num_beams,
            "utterances": selected,
            "rir_manifest": rir_rows,
            "direct_rir_rt60_seconds": direct_rir_rt60_seconds,
            "reverb_rt60_seconds": reverb_rt60_seconds,
            "wpe_backend": "nara_wpe",
            "wpe_config": asdict(WPEConfig()),
            "level_protocol": {
                "target_rms_dbfs": -25.0,
                "peak_headroom_db": 1.0,
            },
            "seed": seed,
        }
    )
    destination = Path(output_path)
    existing = read_jsonl(destination) if destination.is_file() else []
    if any(row.get("run_protocol_sha256") != run_identity for row in existing):
        raise ValueError("existing audit output belongs to another run protocol")
    result_by_key = {
        (str(row["utterance_id"]), str(row["condition"])): row
        for row in existing
    }
    required = {
        (str(row["utterance_id"]), str(condition))
        for row in selected
        for condition in conditions
    }
    resumed = len(required.intersection(result_by_key))
    completed = resumed
    generated = 0
    started = time.perf_counter()
    normalizer = ChineseTextNormalizer()
    corpus = Path(corpus_root)
    rir_directory = Path(rir_root)
    validated_paths: set[Path] = set()

    def notify(
        stage: Literal["start", "progress", "complete"],
        *,
        row: Mapping[str, Any] | None = None,
        condition: AuditCondition | None = None,
        inference_seconds: float | None = None,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            BaselineProgress(
                stage=stage,
                completed=completed,
                total=len(required),
                resumed=resumed,
                generated=generated,
                elapsed_seconds=time.perf_counter() - started,
                utterance_id=None if row is None else str(row["utterance_id"]),
                frontend=None if condition is None else str(condition),
                rt60_seconds=(
                    reverb_rt60_seconds
                    if condition in {"reverb_raw", "reverb_m_wpe_10"}
                    else None
                ),
                inference_seconds=inference_seconds,
            )
        )

    def infer(
        row: Mapping[str, Any],
        waveform: np.ndarray,
        *,
        condition: AuditCondition,
        rir: Mapping[str, Any] | None,
    ) -> None:
        nonlocal completed, generated
        key = (str(row["utterance_id"]), condition)
        if key in result_by_key:
            return
        begin = time.perf_counter()
        hypothesis_raw = transcriber.transcribe(waveform, sample_rate=16_000)
        elapsed = time.perf_counter() - begin
        reference = normalizer.normalize(str(row["transcript"]))
        hypothesis = normalizer.normalize(hypothesis_raw)
        score = score_characters(reference, hypothesis)
        result = {
            "utterance_id": row["utterance_id"],
            "speaker_id": row["speaker_id"],
            "condition": condition,
            "frontend": _condition_frontend(condition),
            "model_id": transcriber.model_id,
            "model_revision": transcriber.model_revision,
            "device": transcriber.device,
            "num_beams": transcriber.num_beams,
            "run_protocol_sha256": run_identity,
            "source_rir_id": None if rir is None else rir["rir_id"],
            "source_room_id": None if rir is None else rir["room_id"],
            "direct_rir_rt60_seconds": (
                direct_rir_rt60_seconds if condition.startswith("direct_") else None
            ),
            "reverb_rt60_seconds": (
                reverb_rt60_seconds
                if condition in {"reverb_raw", "reverb_m_wpe_10"}
                else None
            ),
            "reference_raw": row.get("transcript_raw", row["transcript"]),
            "reference": reference,
            "hypothesis_raw": hypothesis_raw,
            "hypothesis": hypothesis,
            "audio_samples": int(waveform.size),
            "audio_peak_abs": float(np.max(np.abs(waveform), initial=0.0)),
            "audio_rms_dbfs": float(
                20
                * np.log10(
                    max(
                        float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))),
                        np.finfo(float).tiny,
                    )
                )
            ),
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
            condition=condition,
            inference_seconds=elapsed,
        )

    notify("start")
    condition_set = set(conditions)
    direct_conditions = {
        condition for condition in conditions if condition.startswith("direct_")
    }
    reverb_conditions = {
        condition
        for condition in conditions
        if condition in {"reverb_raw", "reverb_m_wpe_10"}
    }
    for row in selected:
        clean = _read_clean(corpus, row)
        if "clean_original" in condition_set:
            infer(row, clean, condition="clean_original", rir=None)
        if "clean_level" in condition_set:
            level = convolve_multichannel(
                clean, np.asarray([[1.0]], dtype=np.float64)
            ).signals[0]
            infer(row, level, condition="clean_level", rir=None)
        if direct_conditions:
            direct_row = _select_rir(
                rir_rows,
                utterance_id=str(row["utterance_id"]),
                rt60_seconds=direct_rir_rt60_seconds,
                seed=seed,
            )
            direct_rir = _load_rir_array(
                rir_directory, direct_row, "direct", validated_paths
            )
            direct_signals = convolve_multichannel(clean, direct_rir).signals
            for condition in conditions:
                if condition not in direct_conditions:
                    continue
                waveform = apply_frontend(
                    direct_signals,
                    _condition_frontend(condition),
                    backend="nara_wpe",
                )
                infer(row, waveform, condition=condition, rir=direct_row)
        if reverb_conditions:
            reverb_row = _select_rir(
                rir_rows,
                utterance_id=str(row["utterance_id"]),
                rt60_seconds=reverb_rt60_seconds,
                seed=seed,
            )
            full_rir = _load_rir_array(
                rir_directory, reverb_row, "full", validated_paths
            )
            reverb_signals = convolve_multichannel(clean, full_rir).signals
            for condition in conditions:
                if condition not in reverb_conditions:
                    continue
                waveform = apply_frontend(
                    reverb_signals,
                    _condition_frontend(condition),
                    backend="nara_wpe",
                )
                infer(row, waveform, condition=condition, rir=reverb_row)

    write_jsonl_atomic(destination, result_by_key.values())
    requested_rows = [
        row for key, row in result_by_key.items() if key in required
    ]
    summary = {
        "schema_version": 1,
        "run_protocol_sha256": run_identity,
        "model_id": transcriber.model_id,
        "model_revision": transcriber.model_revision,
        "device": transcriber.device,
        "num_beams": transcriber.num_beams,
        "utterance_limit": limit,
        "conditions_requested": list(conditions),
        "result_rows": len(requested_rows),
        "resumed_rows": resumed,
        "generated_rows": generated,
        "conditions": _summarize(requested_rows),
        "paired_deltas": _paired_comparisons(
            requested_rows, draws=bootstrap_draws, seed=seed
        ),
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
