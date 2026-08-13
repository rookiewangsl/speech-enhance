"""Apply the robust v2 recovery policy to frozen Whisper v1 hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from speech_frontend.asr.anomaly_detection import diagnose_hypothesis
from speech_frontend.asr.robust_policy import decide_final
from speech_frontend.audio import read_audio

try:
    from scripts.evaluate_asr import (
        WhisperRuntime,
        atomic_write_json,
        atomic_write_jsonl,
        canonical_digest,
        make_whisper_normalizer,
        read_json,
        read_jsonl,
        sha256_file,
        transcribe_options,
        validate_config as validate_v1_config,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from evaluate_asr import (  # type: ignore[no-redef]
        WhisperRuntime,
        atomic_write_json,
        atomic_write_jsonl,
        canonical_digest,
        make_whisper_normalizer,
        read_json,
        read_jsonl,
        sha256_file,
        transcribe_options,
        validate_config as validate_v1_config,
    )


V2_PROTOCOL = "asr_whisper_small_en_robust_v2"
V1_PROTOCOL = "asr_whisper_small_en_v1"
CONDITIONS = ("noisy", "clean", "mcra_dd_wiener", "rnnoise_r3")
Normalizer = Callable[[str], str]
TokenCounter = Callable[[str], int]


class RetryRuntime(Protocol):
    model_sha256: str

    def transcribe(self, samples: Any, options: dict[str, Any]) -> dict[str, Any]: ...

    def synchronize(self) -> None: ...

    def set_seed(self, seed: int) -> None: ...


class SeededWhisperRuntime(WhisperRuntime):
    """Whisper runtime with explicit random seeding for sampled retries."""

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        self._torch.manual_seed(seed)
        if self.device == "mps":
            self._torch.mps.manual_seed(seed)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def retry_runtime_identity(device: str, model_sha256: str) -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "openai_whisper": _package_version("openai-whisper"),
        "torch": _package_version("torch"),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "numpy": _package_version("numpy"),
        "device": device,
        "fp16": False,
        "model_sha256": model_sha256,
    }


def make_token_counter() -> TokenCounter:
    from whisper.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(multilingual=False, language="en", task="transcribe")
    return lambda text: len(tokenizer.encode(str(text)))


def validate_v2_config(config: dict[str, Any], base_config: dict[str, Any]) -> None:
    validate_v1_config(base_config)
    if config.get("schema_version") != 1 or config.get("protocol_version") != V2_PROTOCOL:
        raise ValueError("unsupported robust ASR protocol")
    base = config.get("base_protocol")
    if not isinstance(base, dict):
        raise ValueError("base_protocol must be an object")
    expected = {
        "protocol_version": V1_PROTOCOL,
        "config_digest": canonical_digest(base_config),
        "model_sha256": base_config["model"]["sha256"],
        "device": "cpu",
    }
    if base != expected:
        raise ValueError("robust v2 base_protocol does not match the frozen v1 config")
    detector = config.get("detector")
    if not isinstance(detector, dict):
        raise ValueError("detector must be an object")
    diagnose_hypothesis(
        {
            "hypothesis_normalized": "validation probe",
            "duration_seconds": 1.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 0.9,
                    "compression_ratio": 1.0,
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.1,
                }
            ],
        },
        detector,
    )
    retry = config.get("retry")
    if not isinstance(retry, dict):
        raise ValueError("retry must be an object")
    temperatures = retry.get("temperatures")
    if (
        not isinstance(temperatures, list)
        or not temperatures
        or any(not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0 for value in temperatures)
        or [float(value) for value in temperatures] != sorted({float(value) for value in temperatures})
    ):
        raise ValueError("retry.temperatures must be unique, increasing, and in (0, 1]")
    if not isinstance(retry.get("best_of"), int) or int(retry["best_of"]) <= 0:
        raise ValueError("retry.best_of must be a positive integer")
    if not isinstance(retry.get("seed_namespace"), str) or not retry["seed_namespace"]:
        raise ValueError("retry.seed_namespace must be non-empty")
    if not isinstance(retry.get("warmup"), bool):
        raise ValueError("retry.warmup must be boolean")
    fallback = config.get("fallback")
    if not isinstance(fallback, dict) or fallback.get("source_condition") != "noisy":
        raise ValueError("the only permitted fallback source is noisy")
    eligible = fallback.get("eligible_conditions")
    if set(eligible or []) != {"mcra_dd_wiener", "rnnoise_r3"}:
        raise ValueError("fallback is frozen to the two enhanced conditions")
    if config.get("abstention") != {"enabled": True}:
        raise ValueError("robust v2 must enable abstention")
    selection = config.get("selection")
    if not isinstance(selection, dict) or selection.get("enhanced_retry_requires_noisy_consistency") is not True:
        raise ValueError("robust v2 must gate enhanced retries against the noisy path")
    consistency_threshold = selection.get("max_retry_noisy_word_distance")
    if not isinstance(consistency_threshold, (int, float)) or not 0.0 <= float(consistency_threshold) <= 1.0:
        raise ValueError("selection.max_retry_noisy_word_distance must be in [0, 1]")
    if config.get("normalization") != base_config.get("normalization"):
        raise ValueError("v2 normalization must match v1")


def _validate_v1_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    if not rows:
        raise ValueError("v1 hypotheses are empty")
    expected_digest = str(config["base_protocol"]["config_digest"])
    expected_model = str(config["base_protocol"]["model_sha256"])
    expected_device = str(config["base_protocol"]["device"])
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        utterance_id = str(row.get("id", "")).strip()
        condition = str(row.get("condition", "")).strip()
        if not utterance_id or condition not in CONDITIONS:
            raise ValueError("each v1 row requires a valid id and condition")
        if condition in grouped[utterance_id]:
            raise ValueError(f"duplicate v1 row: {utterance_id}/{condition}")
        if row.get("status") != "completed":
            raise ValueError(f"v1 row is not completed: {utterance_id}/{condition}")
        if row.get("asr_config_digest") != expected_digest:
            raise ValueError("v1 row ASR config digest does not match robust base protocol")
        if row.get("model_sha256") != expected_model or row.get("device") != expected_device:
            raise ValueError("v1 row model/device identity does not match robust base protocol")
        grouped[utterance_id][condition] = row
    for utterance_id, condition_rows in grouped.items():
        if set(condition_rows) != set(CONDITIONS):
            raise ValueError(f"v1 utterance is not four-way paired: {utterance_id}")
        reference_hashes = {row.get("reference_raw_sha256") for row in condition_rows.values()}
        if len(reference_hashes) != 1:
            raise ValueError(f"paired reference mismatch: {utterance_id}")
        normalized_references = {row.get("reference_normalized") for row in condition_rows.values()}
        if len(normalized_references) != 1:
            raise ValueError(f"paired normalized reference mismatch: {utterance_id}")
    return dict(grouped)


def _seed(namespace: str, utterance_id: str, condition: str, temperature: float) -> int:
    payload = f"{namespace}\0{utterance_id}\0{condition}\0{temperature:.6f}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _segment_metadata(result: dict[str, Any], expected_temperature: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        if not isinstance(segment, dict):
            raise RuntimeError("Whisper retry returned non-object segment metadata")
        value = {
            name: segment.get(name)
            for name in (
                "id",
                "start",
                "end",
                "text",
                "temperature",
                "avg_logprob",
                "compression_ratio",
                "no_speech_prob",
            )
        }
        for field in ("start", "end", "temperature", "avg_logprob", "compression_ratio", "no_speech_prob"):
            number = value.get(field)
            if not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                raise RuntimeError(f"Whisper retry segment has invalid {field}")
        if not math.isclose(float(value["temperature"]), expected_temperature, abs_tol=1e-9):
            raise RuntimeError("Whisper retry used an unexpected temperature")
        output.append(value)
    return output


def _first_pass_attempt(row: dict[str, Any], detector: dict[str, Any], token_counter: TokenCounter) -> dict[str, Any]:
    attempt = {
        "attempt_kind": "first_pass",
        "source_condition": row["condition"],
        "temperature": 0.0,
        "seed": None,
        "hypothesis_raw": row["hypothesis_raw"],
        "hypothesis_normalized": row["hypothesis_normalized"],
        "segments": row["segments"],
        "asr_seconds": float(row["asr_seconds"]),
        "audio_sha256": row["audio_sha256"],
    }
    attempt["diagnostic"] = diagnose_hypothesis(attempt | {"duration_seconds": row["duration_seconds"]}, detector, token_counter=token_counter)
    return attempt


def _retry_options(base_config: dict[str, Any], temperature: float, best_of: int) -> dict[str, Any]:
    options = transcribe_options(base_config)
    options.pop("beam_size", None)
    options.pop("patience", None)
    options["temperature"] = temperature
    options["best_of"] = best_of
    return options


def _cache_key(
    row: dict[str, Any],
    noisy_row: dict[str, Any],
    *,
    policy_digest: str,
    evaluator_sha256: str,
    retry_runtime_digest: str,
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "condition": row["condition"],
        "v1_row_digest": canonical_digest(row),
        "v1_noisy_row_digest": canonical_digest(noisy_row),
        "policy_config_digest": policy_digest,
        "robust_evaluator_sha256": evaluator_sha256,
        "retry_runtime_identity_digest": retry_runtime_digest,
    }


def _valid_cache(path: Path, key: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if any(value.get(field) != expected for field, expected in key.items()):
        return None
    if value.get("final_status") not in {"accepted", "abstained"}:
        return None
    return value


def evaluate_robust(
    *,
    v1_rows: list[dict[str, Any]],
    v1_hypotheses_sha256: str,
    config: dict[str, Any],
    base_config: dict[str, Any],
    project_root: Path,
    output_path: Path,
    cache_root: Path,
    environment_output: Path,
    retry_runtime_identity_value: dict[str, Any],
    runtime_factory: Callable[[], RetryRuntime],
    normalizer: Normalizer,
    token_counter: TokenCounter,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Apply v2 and atomically write one final record per v1 record."""

    validate_v2_config(config, base_config)
    grouped = _validate_v1_rows(v1_rows, config)
    policy_digest = canonical_digest(config)
    evaluator_sha256 = sha256_file(Path(__file__).resolve())
    retry_runtime_digest = canonical_digest(retry_runtime_identity_value)
    detector = config["detector"]
    retry_config = config["retry"]
    fallback_conditions = set(config["fallback"]["eligible_conditions"])
    abstention_enabled = bool(config["abstention"]["enabled"])
    consistency_threshold = float(config["selection"]["max_retry_noisy_word_distance"])

    runtime: RetryRuntime | None = None
    warmed_up = False
    model_loads = retry_inferences = retry_cache_hits = final_cache_hits = 0

    def get_runtime() -> RetryRuntime:
        nonlocal runtime, model_loads
        if runtime is None:
            runtime = runtime_factory()
            model_loads += 1
            if runtime.model_sha256 != config["base_protocol"]["model_sha256"]:
                raise RuntimeError("retry runtime model SHA-256 mismatch")
        return runtime

    results: list[dict[str, Any]] = []
    final_by_id_condition: dict[tuple[str, str], dict[str, Any]] = {}
    for utterance_id in sorted(grouped):
        rows = grouped[utterance_id]
        for condition in CONDITIONS:
            row = rows[condition]
            key = _cache_key(
                row,
                rows["noisy"],
                policy_digest=policy_digest,
                evaluator_sha256=evaluator_sha256,
                retry_runtime_digest=retry_runtime_digest,
            )
            cache_path = cache_root / "final" / condition / f"{utterance_id}.json"
            cached = None if force else _valid_cache(cache_path, key)
            if cached is not None:
                results.append(cached)
                final_by_id_condition[(utterance_id, condition)] = cached
                final_cache_hits += 1
                continue

            first_pass = _first_pass_attempt(row, detector, token_counter)
            attempts: list[dict[str, Any]] = []
            if first_pass["diagnostic"]["anomalous"]:
                audio_path = Path(str(row["audio"]))
                if not audio_path.is_absolute():
                    audio_path = project_root / audio_path
                if sha256_file(audio_path) != row["audio_sha256"]:
                    raise RuntimeError(f"retry audio SHA-256 mismatch: {utterance_id}/{condition}")
                audio = read_audio(audio_path)
                if audio.sample_rate != int(base_config["model"]["sample_rate_hz"]):
                    raise RuntimeError("retry audio sample rate mismatch")
                if audio.samples.size != int(row["num_samples"]):
                    raise RuntimeError("retry audio sample count mismatch")

                for temperature_value in retry_config["temperatures"]:
                    temperature = float(temperature_value)
                    seed = _seed(str(retry_config["seed_namespace"]), utterance_id, condition, temperature)
                    attempt_key = {
                        **key,
                        "temperature": temperature,
                        "seed": seed,
                        "audio_sha256": row["audio_sha256"],
                    }
                    attempt_path = cache_root / "attempts" / condition / utterance_id / f"temperature_{temperature:.1f}.json"
                    cached_attempt = None if force else _valid_cache(attempt_path, attempt_key)
                    if cached_attempt is not None:
                        attempt = cached_attempt
                        retry_cache_hits += 1
                    else:
                        active_runtime = get_runtime()
                        if bool(retry_config["warmup"]) and not warmed_up:
                            warmup_options = transcribe_options(base_config)
                            active_runtime.set_seed(0)
                            active_runtime.transcribe(np.zeros(16_000, dtype=np.float32), warmup_options)
                            active_runtime.synchronize()
                            warmed_up = True
                        active_runtime.set_seed(seed)
                        options = _retry_options(base_config, temperature, int(retry_config["best_of"]))
                        active_runtime.synchronize()
                        started = time.perf_counter()
                        transcribed = active_runtime.transcribe(audio.samples, options)
                        active_runtime.synchronize()
                        elapsed = time.perf_counter() - started
                        hypothesis_raw = str(transcribed.get("text", "")).strip()
                        attempt = {
                            **attempt_key,
                            "final_status": "accepted",
                            "attempt_kind": "temperature_retry",
                            "source_condition": condition,
                            "hypothesis_raw": hypothesis_raw,
                            "hypothesis_normalized": normalizer(hypothesis_raw),
                            "segments": _segment_metadata(transcribed, temperature),
                            "asr_seconds": elapsed,
                        }
                        attempt["diagnostic"] = diagnose_hypothesis(
                            attempt | {"duration_seconds": row["duration_seconds"]},
                            detector,
                            token_counter=token_counter,
                        )
                        atomic_write_json(attempt_path, attempt)
                        retry_inferences += 1
                    attempts.append(attempt)
                    if attempt["diagnostic"]["anomalous"] is False:
                        break

            noisy_result = final_by_id_condition.get((utterance_id, "noisy"))
            decision = decide_final(
                condition=condition,
                first_pass=first_pass,
                retry_attempts=attempts,
                fallback_conditions=fallback_conditions,
                noisy_result=noisy_result,
                abstention_enabled=abstention_enabled,
                retry_noisy_consistency_threshold=consistency_threshold,
            )
            selected = decision["attempt"]
            retry_seconds = sum(float(attempt["asr_seconds"]) for attempt in attempts)
            fallback_seconds = (
                float(noisy_result["total_service_asr_seconds"])
                if decision["source"] == "noisy_fallback" and noisy_result is not None
                else 0.0
            )
            total_asr_seconds = float(first_pass["asr_seconds"]) + retry_seconds + fallback_seconds
            processing_seconds = float(row.get("processing_seconds", 0.0))
            duration = float(row["duration_seconds"])
            final = {
                **row,
                **key,
                "schema_version": 1,
                "protocol_version": V2_PROTOCOL,
                "v1_hypotheses_sha256": v1_hypotheses_sha256,
                "status": "completed" if decision["status"] == "accepted" else "abstained",
                "first_pass_attempt": first_pass,
                "first_pass_anomalous": bool(first_pass["diagnostic"]["anomalous"]),
                "trigger_reasons": list(first_pass["diagnostic"]["reasons"]),
                "retry_attempts": attempts,
                "final_status": decision["status"],
                "final_source": decision["source"],
                "final_attempt": selected,
                "decision_details": decision.get("decision_details", {}),
                "abstained": decision["status"] == "abstained",
                "hypothesis_raw": "" if selected is None else selected["hypothesis_raw"],
                "hypothesis_normalized": "" if selected is None else selected["hypothesis_normalized"],
                "segments": [] if selected is None else selected["segments"],
                "first_pass_asr_seconds": float(first_pass["asr_seconds"]),
                "recovery_asr_seconds": retry_seconds + fallback_seconds,
                "total_service_asr_seconds": total_asr_seconds,
                "asr_seconds": total_asr_seconds,
                "asr_rtf": total_asr_seconds / duration,
                "end_to_end_seconds": processing_seconds + total_asr_seconds,
                "end_to_end_rtf": (processing_seconds + total_asr_seconds) / duration,
            }
            atomic_write_json(cache_path, final)
            results.append(final)
            final_by_id_condition[(utterance_id, condition)] = final
            print(
                f"{utterance_id}/{condition}: {decision['source']}, "
                f"triggered={first_pass['diagnostic']['anomalous']}",
                flush=True,
            )

    results.sort(key=lambda row: (str(row["id"]), CONDITIONS.index(str(row["condition"]))))
    source_counts = Counter(str(row["final_source"]) for row in results)
    reason_counts = Counter(reason for row in results for reason in row["trigger_reasons"])
    environment = {
        "protocol_version": V2_PROTOCOL,
        "policy_config_digest": policy_digest,
        "robust_evaluator_sha256": evaluator_sha256,
        "base_hypotheses_sha256": v1_hypotheses_sha256,
        "retry_runtime_identity": retry_runtime_identity_value,
        "retry_runtime_identity_digest": retry_runtime_digest,
        "rows": len(results),
        "first_pass_triggers": sum(bool(row["first_pass_anomalous"]) for row in results),
        "trigger_reason_counts": dict(sorted(reason_counts.items())),
        "final_source_counts": dict(sorted(source_counts.items())),
        "abstentions": sum(bool(row["abstained"]) for row in results),
        "final_cache_hits": final_cache_hits,
        "retry_cache_hits": retry_cache_hits,
        "retry_inferences": retry_inferences,
        "model_loads": model_loads,
    }
    atomic_write_json(environment_output, environment)
    atomic_write_jsonl(output_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-hypotheses", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/asr_whisper_small_en_robust_v2.json"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/asr_whisper_small_en.json"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    base_config = read_json(args.base_config)
    validate_v2_config(config, base_config)
    v1_path = args.v1_hypotheses.resolve()
    identity = retry_runtime_identity(args.device, str(base_config["model"]["sha256"]))
    evaluate_robust(
        v1_rows=read_jsonl(v1_path),
        v1_hypotheses_sha256=sha256_file(v1_path),
        config=config,
        base_config=base_config,
        project_root=args.project_root.resolve(),
        output_path=args.output,
        cache_root=args.cache_root,
        environment_output=args.environment_output,
        retry_runtime_identity_value=identity,
        runtime_factory=lambda: SeededWhisperRuntime(base_config, args.model_root.resolve(), args.device),
        normalizer=make_whisper_normalizer(),
        token_counter=make_token_counter(),
        force=args.force,
    )


if __name__ == "__main__":
    main()
