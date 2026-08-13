"""Run frozen local Whisper on cached ASR inputs with resumable results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from speech_frontend.audio import read_audio


SUCCESS_STATUS = "completed"
FROZEN_PROTOCOL_VERSION = "asr_whisper_small_en_v1"
FROZEN_MODEL_SHA256 = (
    "f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872"
)
Normalizer = Callable[[str], str]


class ASRRuntime(Protocol):
    """Minimal inference interface used by the real runtime and tests."""

    model_sha256: str

    def transcribe(self, samples: Any, options: dict[str, Any]) -> dict[str, Any]: ...

    def synchronize(self) -> None: ...

    def environment(self) -> dict[str, Any]: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL contains no rows: {path}")
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(
        path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported ASR config schema_version")
    if config.get("protocol_version") != FROZEN_PROTOCOL_VERSION:
        raise ValueError("unexpected frozen ASR protocol_version")
    model = config.get("model", {})
    if model.get("name") != "small.en":
        raise ValueError("frozen ASR model must be small.en")
    expected_sha = str(model.get("sha256", ""))
    if expected_sha != FROZEN_MODEL_SHA256:
        raise ValueError("model.sha256 does not match frozen small.en weight")
    if model.get("sample_rate_hz") != 16_000:
        raise ValueError("frozen ASR sample rate must be 16 kHz")
    decoding = config.get("decoding", {})
    expected = {
        "task": "transcribe",
        "language": "en",
        "temperature": 0.0,
        "beam_size": 5,
        "patience": 1.0,
        "condition_on_previous_text": False,
        "fp16": False,
    }
    for key, value in expected.items():
        if decoding.get(key) != value:
            raise ValueError(f"frozen decoding value mismatch: {key}={decoding.get(key)!r}")
    if decoding.get("best_of") is not None:
        raise ValueError("best_of must be null for deterministic beam search")
    if decoding.get("initial_prompt") is not None:
        raise ValueError("initial_prompt must be null")
    if decoding.get("length_penalty") is not None:
        raise ValueError("length_penalty must be null")
    if decoding.get("verbose") is not None:
        raise ValueError("verbose must be null")
    expected_thresholds = {
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }
    if config.get("thresholds") != expected_thresholds:
        raise ValueError("Whisper thresholds do not match frozen protocol")
    if config.get("normalization") != {
        "implementation": "whisper.normalizers.EnglishTextNormalizer"
    }:
        raise ValueError("normalization does not match frozen protocol")
    if config.get("timing") != {
        "warmup": True,
        "exclude_audio_read": True,
        "exclude_model_load": True,
        "exclude_result_write": True,
    }:
        raise ValueError("timing contract does not match frozen protocol")


def transcribe_options(config: dict[str, Any]) -> dict[str, Any]:
    options = dict(config["decoding"])
    options.update(config["thresholds"])
    return options


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


class WhisperRuntime:
    """Official openai-whisper runtime with package and weight verification."""

    def __init__(self, config: dict[str, Any], model_root: Path, device: str) -> None:
        import torch
        import whisper

        expected_whisper = str(config["implementation"]["version"])
        actual_whisper = _package_version("openai-whisper")
        if actual_whisper != expected_whisper:
            raise RuntimeError(
                f"openai-whisper version mismatch: {actual_whisper} != {expected_whisper}"
            )
        expected_torch = str(config["implementation"]["torch_version"])
        actual_torch = _package_version("torch")
        if actual_torch != expected_torch:
            raise RuntimeError(f"torch version mismatch: {actual_torch} != {expected_torch}")
        if device not in {"cpu", "mps"}:
            raise ValueError("device must be cpu or mps")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")

        model_root.mkdir(parents=True, exist_ok=True)
        model_path = model_root / f"{config['model']['name']}.pt"
        if (
            model_path.is_file()
            and sha256_file(model_path) != config["model"]["sha256"]
        ):
            raise RuntimeError(
                "existing Whisper model SHA-256 does not match frozen config"
            )
        self.device = device
        self._torch = torch
        self._whisper = whisper
        self._model = whisper.load_model(
            str(config["model"]["name"]),
            device=device,
            download_root=str(model_root),
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"Whisper model file not found: {model_path}")
        self.model_sha256 = sha256_file(model_path)
        if self.model_sha256 != config["model"]["sha256"]:
            raise RuntimeError("Whisper model SHA-256 does not match frozen config")
        self.model_path = model_path.resolve()

    def transcribe(self, samples: Any, options: dict[str, Any]) -> dict[str, Any]:
        return self._model.transcribe(samples, **options)

    def synchronize(self) -> None:
        if self.device == "mps":
            self._torch.mps.synchronize()

    def environment(self) -> dict[str, Any]:
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "openai_whisper": _package_version("openai-whisper"),
            "torch": _package_version("torch"),
            "torch_num_threads": int(self._torch.get_num_threads()),
            "torch_num_interop_threads": int(self._torch.get_num_interop_threads()),
            "numpy": _package_version("numpy"),
            "ffmpeg": _ffmpeg_version(),
            "device": self.device,
            "fp16": False,
            "mps_available": bool(self._torch.backends.mps.is_available()),
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
        }


def _ffmpeg_version() -> str:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.splitlines()[0] if result.stdout else "unknown"


def _git_revision(project_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def make_whisper_normalizer() -> Normalizer:
    from whisper.normalizers import EnglishTextNormalizer

    return EnglishTextNormalizer()


def _resolve_audio(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _index_references(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        utterance_id = str(row.get("id", "")).strip()
        reference = row.get("reference_raw")
        if not utterance_id or not isinstance(reference, str) or not reference.strip():
            raise ValueError("each reference needs non-empty id and reference_raw")
        if utterance_id in indexed:
            raise ValueError(f"duplicate reference id: {utterance_id}")
        indexed[utterance_id] = row
    return indexed


def _validate_inputs(rows: list[dict[str, Any]], references: dict[str, Any]) -> None:
    keys: set[tuple[str, str]] = set()
    ids_by_condition: dict[str, set[str]] = {}
    reserved = {
        "status",
        "reference_raw",
        "reference_normalized",
        "reference_raw_sha256",
        "hypothesis_raw",
        "hypothesis_normalized",
        "segments",
        "asr_seconds",
        "asr_rtf",
        "end_to_end_seconds",
        "end_to_end_rtf",
        "model_sha256",
        "asr_config_digest",
        "evaluator_code_sha256",
        "runtime_identity_digest",
        "device",
    }
    for row in rows:
        utterance_id = str(row.get("id", "")).strip()
        condition = str(row.get("condition", "")).strip()
        if not utterance_id or not condition or not row.get("audio"):
            raise ValueError("each ASR input needs id, condition, and audio")
        forbidden = reserved & row.keys()
        if forbidden:
            raise ValueError(
                f"ASR input contains reserved result fields: {sorted(forbidden)}"
            )
        key = (utterance_id, condition)
        if key in keys:
            raise ValueError(f"duplicate ASR input key: {key}")
        keys.add(key)
        if utterance_id not in references:
            raise ValueError(f"ASR input has no reference: {utterance_id}")
        ids_by_condition.setdefault(condition, set()).add(utterance_id)
    if not ids_by_condition:
        raise ValueError("no ASR inputs")
    expected = next(iter(ids_by_condition.values()))
    for condition, actual in ids_by_condition.items():
        if actual != expected:
            raise ValueError(f"condition {condition} does not contain the paired ID set")


def _cache_key(
    row: dict[str, Any],
    reference_raw: str,
    config_digest: str,
    model_sha256: str,
    evaluator_code_sha256: str,
    runtime_identity_digest: str,
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "condition": row["condition"],
        "audio_sha256": row["audio_sha256"],
        "reference_raw_sha256": hashlib.sha256(reference_raw.encode()).hexdigest(),
        "asr_config_digest": config_digest,
        "model_sha256": model_sha256,
        "evaluator_code_sha256": evaluator_code_sha256,
        "runtime_identity_digest": runtime_identity_digest,
    }


def _valid_cache(path: Path, key: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if value.get("status") != SUCCESS_STATUS:
        return None
    if any(value.get(name) != expected for name, expected in key.items()):
        return None
    required = {"hypothesis_raw", "hypothesis_normalized", "reference_normalized"}
    return value if required <= value.keys() else None


def _refresh_cached_result(
    cached: dict[str, Any], input_row: dict[str, Any], key: dict[str, Any]
) -> dict[str, Any]:
    """Combine immutable ASR cache data with current front-end metadata."""

    duration = float(input_row["duration_seconds"])
    processing_seconds = float(input_row.get("processing_seconds", 0.0))
    asr_seconds = float(cached["asr_seconds"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    if not math.isfinite(processing_seconds) or processing_seconds < 0:
        raise ValueError("processing_seconds must be finite and non-negative")
    if not math.isfinite(asr_seconds) or asr_seconds < 0:
        raise ValueError("cached asr_seconds must be finite and non-negative")
    # Current input provenance wins; ASR output and its frozen identity remain
    # from the validated cache.  This prevents stale enhancement metadata and
    # end-to-end timing when identical audio is re-exported by newer code.
    result = {**cached, **input_row, **key}
    result.update(
        {
            "status": SUCCESS_STATUS,
            "device": cached["device"],
            "runtime_identity_digest": cached["runtime_identity_digest"],
            "asr_seconds": asr_seconds,
            "asr_rtf": asr_seconds / duration,
            "end_to_end_seconds": processing_seconds + asr_seconds,
            "end_to_end_rtf": (processing_seconds + asr_seconds) / duration,
        }
    )
    return result


def _segment_metadata(result: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        output.append(
            {
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
        )
    return output


def _validate_segments(segments: list[dict[str, Any]]) -> None:
    for segment in segments:
        for field in (
            "start",
            "end",
            "temperature",
            "avg_logprob",
            "compression_ratio",
            "no_speech_prob",
        ):
            value = segment.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"Whisper segment has invalid {field}")
        if float(segment["temperature"]) != 0.0:
            raise RuntimeError("Whisper used a non-zero segment temperature")
        if float(segment["end"]) < float(segment["start"]):
            raise RuntimeError("Whisper segment end precedes start")
        probability = float(segment["no_speech_prob"])
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError("Whisper segment no_speech_prob is outside [0, 1]")


def evaluate(
    *,
    input_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    project_root: Path,
    output_path: Path,
    cache_root: Path,
    environment_output: Path,
    config: dict[str, Any],
    runtime: ASRRuntime,
    normalizer: Normalizer,
    force: bool = False,
    warmup: bool = True,
) -> list[dict[str, Any]]:
    """Transcribe paired inputs and atomically materialize the final JSONL."""

    validate_config(config)
    reference_by_id = _index_references(reference_rows)
    _validate_inputs(input_rows, reference_by_id)
    digest = canonical_digest(config)
    evaluator_code_sha256 = sha256_file(Path(__file__).resolve())
    runtime_environment = runtime.environment()
    runtime_identity = {
        name: runtime_environment[name]
        for name in (
            "python",
            "platform",
            "machine",
            "openai_whisper",
            "torch",
            "torch_num_threads",
            "torch_num_interop_threads",
            "numpy",
            "device",
            "fp16",
            "model_sha256",
        )
        if name in runtime_environment
    }
    if not isinstance(runtime_identity.get("device"), str):
        raise ValueError("runtime environment must identify its device")
    runtime_identity_digest = canonical_digest(runtime_identity)
    options = transcribe_options(config)

    results: list[dict[str, Any]] = []
    warmed_up = False
    cache_hits = 0
    transcribed_rows = 0
    for input_row in input_rows:
        utterance_id = str(input_row["id"])
        condition = str(input_row["condition"])
        reference_row = reference_by_id[utterance_id]
        reference_raw = str(reference_row["reference_raw"])
        key = _cache_key(
            input_row,
            reference_raw,
            digest,
            runtime.model_sha256,
            evaluator_code_sha256,
            runtime_identity_digest,
        )
        cache_path = cache_root / condition / f"{utterance_id}.json"
        cached = None if force else _valid_cache(cache_path, key)
        if cached is not None:
            results.append(_refresh_cached_result(cached, input_row, key))
            cache_hits += 1
            continue

        audio_path = _resolve_audio(project_root, str(input_row["audio"]))
        audio = read_audio(audio_path)
        if audio.sample_rate != int(config["model"]["sample_rate_hz"]):
            raise ValueError(f"ASR audio must be 16 kHz: {audio_path}")
        if audio.samples.size != int(input_row["num_samples"]):
            raise ValueError(f"ASR input length mismatch: {audio_path}")
        if sha256_file(audio_path) != input_row["audio_sha256"]:
            raise ValueError(f"ASR input checksum mismatch: {audio_path}")
        duration = audio.samples.size / audio.sample_rate
        front_end_seconds = float(input_row.get("processing_seconds", 0.0))
        if not math.isfinite(front_end_seconds) or front_end_seconds < 0:
            raise ValueError("processing_seconds must be finite and non-negative")

        # Delay warm-up until the first cache miss.  A fully cached resume then
        # performs zero transcriptions, while a partial resume still warms the
        # backend immediately before its first timed inference.
        if warmup and not warmed_up:
            runtime.transcribe(audio.samples, options)
            runtime.synchronize()
            warmed_up = True

        runtime.synchronize()
        started = time.perf_counter()
        try:
            transcribed = runtime.transcribe(audio.samples, options)
            runtime.synchronize()
            segments = _segment_metadata(transcribed)
            _validate_segments(segments)
        except BaseException as error:
            failure = {
                **key,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            atomic_write_json(cache_path, failure)
            raise
        asr_seconds = time.perf_counter() - started
        hypothesis_raw = str(transcribed.get("text", "")).strip()
        result = {
            **input_row,
            **key,
            "status": SUCCESS_STATUS,
            "device": runtime_identity["device"],
            "runtime_identity_digest": runtime_identity_digest,
            "sample_rate": audio.sample_rate,
            "num_samples": int(audio.samples.size),
            "duration_seconds": duration,
            "reference_raw": reference_raw,
            "reference_normalized": normalizer(reference_raw),
            "reference_metadata": {
                name: value
                for name, value in reference_row.items()
                if name not in {"id", "reference_raw"}
            },
            "hypothesis_raw": hypothesis_raw,
            "hypothesis_normalized": normalizer(hypothesis_raw),
            "asr_seconds": asr_seconds,
            "asr_rtf": asr_seconds / duration,
            "end_to_end_seconds": front_end_seconds + asr_seconds,
            "end_to_end_rtf": (front_end_seconds + asr_seconds) / duration,
            "segments": segments,
        }
        atomic_write_json(cache_path, result)
        results.append(result)
        transcribed_rows += 1
        print(
            f"{utterance_id}/{condition}: ASR RTF={result['asr_rtf']:.3f}",
            flush=True,
        )

    environment = {
        **runtime_environment,
        "runtime_identity": runtime_identity,
        "runtime_identity_digest": runtime_identity_digest,
        "protocol_version": config["protocol_version"],
        "git_revision": _git_revision(project_root),
        "asr_config_digest": digest,
        "evaluator_code_sha256": evaluator_code_sha256,
        "decoding": options,
        "normalization": config["normalization"],
        "utterance_condition_rows": len(results),
        "cache_hits": cache_hits,
        "transcribed_rows": transcribed_rows,
    }
    atomic_write_json(environment_output, environment)
    atomic_write_jsonl(output_path, results)
    return results


def _limit_utterances(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    if limit <= 0:
        raise ValueError("limit-utterances must be positive")
    selected: list[str] = []
    for row in rows:
        utterance_id = str(row["id"])
        if utterance_id not in selected:
            selected.append(utterance_id)
        if len(selected) == limit:
            break
    selected_set = set(selected)
    return [row for row in rows if str(row["id"]) in selected_set]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/asr_whisper_small_en.json"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--limit-utterances", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    arguments = parser.parse_args()

    project_root = arguments.project_root.resolve()
    config = read_json(arguments.config)
    validate_config(config)
    inputs = _limit_utterances(read_jsonl(arguments.inputs), arguments.limit_utterances)
    references = read_jsonl(arguments.references)
    runtime = WhisperRuntime(config, arguments.model_root.resolve(), arguments.device)
    evaluate(
        input_rows=inputs,
        reference_rows=references,
        project_root=project_root,
        output_path=arguments.output,
        cache_root=arguments.cache_root,
        environment_output=arguments.environment_output,
        config=config,
        runtime=runtime,
        normalizer=make_whisper_normalizer(),
        force=arguments.force,
        warmup=not arguments.no_warmup,
    )


if __name__ == "__main__":
    main()
