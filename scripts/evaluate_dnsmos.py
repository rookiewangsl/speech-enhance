"""Score an audio-input JSONL with frozen local DNSMOS P.835."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import speech_frontend.dnsmos as dnsmos_implementation
from speech_frontend.audio import read_audio
from speech_frontend.dnsmos import (
    canonical_json_digest,
    load_protocol,
    score_audio,
    sha256_file,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def create_session(model_path: Path, config: dict[str, Any]) -> tuple[Any, str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError("install the project perceptual extra to run DNSMOS") from exc
    expected_version = str(config["implementation"]["onnxruntime_version"])
    if ort.__version__ != expected_version:
        raise RuntimeError(
            f"onnxruntime version mismatch: expected {expected_version}, found {ort.__version__}"
        )
    runtime = config["runtime"]
    if runtime["execution_mode"] != "sequential":
        raise ValueError("frozen DNSMOS runtime requires sequential execution")
    if runtime["graph_optimization_level"] != "all":
        raise ValueError("unsupported DNSMOS graph optimization level")
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(runtime["intra_op_num_threads"])
    options.inter_op_num_threads = int(runtime["inter_op_num_threads"])
    if options.intra_op_num_threads <= 0 or options.inter_op_num_threads <= 0:
        raise ValueError("DNSMOS ONNX thread counts must be positive")
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=[str(runtime["provider"])],
    )
    return session, ort.__version__


def validate_session(session: Any, config: dict[str, Any]) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    expected_input = config["model"]["input_name"]
    expected_output = config["model"]["output_name"]
    expected_samples = int(config["audio"]["input_samples"])
    if (
        len(inputs) != 1
        or inputs[0].name != expected_input
        or inputs[0].shape != ["N", expected_samples]
        or inputs[0].type != "tensor(float)"
    ):
        raise ValueError("DNSMOS ONNX input contract does not match frozen config")
    if (
        len(outputs) != 1
        or outputs[0].name != expected_output
        or outputs[0].shape != ["N", 3]
        or outputs[0].type != "tensor(float)"
    ):
        raise ValueError("DNSMOS ONNX output contract does not match frozen config")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/dnsmos_p835.json"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    project_root = args.project_root.resolve()
    config = read_json(project_path(project_root, args.config))
    protocol = load_protocol(config)
    config_digest = canonical_json_digest(config)
    evaluator_sha = sha256_file(Path(__file__).resolve())
    implementation_sha = sha256_file(Path(dnsmos_implementation.__file__).resolve())
    model_path = args.model_root / str(config["model"]["name"])
    if not model_path.is_file():
        raise FileNotFoundError(f"DNSMOS model not found: {model_path}")
    actual_model_sha = sha256_file(model_path)
    if actual_model_sha != protocol.model_sha256:
        raise ValueError(f"DNSMOS model SHA mismatch: {actual_model_sha}")

    rows = read_jsonl(args.inputs)
    if args.limit is not None:
        rows = rows[: args.limit]
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("DNSMOS input must have a non-empty string id")
        if not isinstance(row.get("condition"), str) or not row["condition"]:
            raise ValueError("DNSMOS input must have a non-empty string condition")
        identity = (row["id"], row["condition"])
        if identity in seen:
            raise ValueError(f"invalid or duplicate DNSMOS input identity: {identity}")
        if not row.get("audio"):
            raise ValueError(f"DNSMOS input missing audio path: {identity}")
        seen.add(identity)

    args.cache_root.mkdir(parents=True, exist_ok=True)
    session: Any | None = None
    runtime_version = str(config["implementation"]["onnxruntime_version"])
    output_rows: list[dict[str, Any]] = []
    cache_hits = 0
    scored_rows = 0
    for index, input_row in enumerate(rows, start=1):
        audio_path = project_path(project_root, str(input_row["audio"]))
        actual_audio_sha = sha256_file(audio_path)
        declared_audio_sha = input_row.get("audio_sha256")
        if declared_audio_sha is not None and declared_audio_sha != actual_audio_sha:
            raise ValueError(f"audio SHA mismatch: {audio_path}")
        identity = {
            "id": input_row["id"],
            "condition": input_row["condition"],
            "audio_sha256": actual_audio_sha,
            "model_sha256": actual_model_sha,
            "config_digest": config_digest,
            "evaluator_sha256": evaluator_sha,
            "implementation_sha256": implementation_sha,
            "onnxruntime_version": runtime_version,
        }
        cache_key = canonical_json_digest(identity)
        cache_path = args.cache_root / f"{cache_key}.json"
        if cache_path.exists() and not args.force:
            cached = read_json(cache_path)
            if cached.get("cache_identity") != identity:
                raise ValueError(f"DNSMOS cache identity mismatch: {cache_path}")
            output_rows.append(cached["result"])
            cache_hits += 1
            continue
        if session is None:
            session, runtime_version = create_session(model_path, config)
            validate_session(session, config)
        audio = read_audio(audio_path)
        started = time.perf_counter()
        scores = score_audio(audio, session, protocol)
        elapsed = time.perf_counter() - started
        result = {
            "id": input_row["id"],
            "condition": input_row["condition"],
            "speaker_id": input_row.get("speaker_id", "unknown"),
            "split": input_row.get("split", "unknown"),
            "noise": input_row.get("noise", "unknown"),
            "snr_db": input_row.get("snr_db", "unknown"),
            "audio": str(input_row["audio"]),
            "audio_sha256": actual_audio_sha,
            "protocol_version": protocol.protocol_version,
            "config_digest": config_digest,
            "model_sha256": actual_model_sha,
            **scores,
            "processing_seconds": elapsed,
            "rtf_processing_only": elapsed / float(scores["duration_seconds"]),
        }
        atomic_json(cache_path, {"cache_identity": identity, "result": result})
        output_rows.append(result)
        scored_rows += 1
        print(
            f"[{index}/{len(rows)}] {result['id']} {result['condition']}: "
            f"SIG={result['sig']:.3f} BAK={result['bak']:.3f} OVRL={result['ovrl']:.3f}",
            flush=True,
        )

    atomic_jsonl(args.output, output_rows)
    environment = {
        "schema_version": 1,
        "protocol_version": protocol.protocol_version,
        "rows": len(output_rows),
        "cache_hits": cache_hits,
        "scored_rows": scored_rows,
        "config_digest": config_digest,
        "model_path": str(model_path),
        "model_sha256": actual_model_sha,
        "evaluator_sha256": evaluator_sha,
        "implementation_sha256": implementation_sha,
        "onnxruntime_version": runtime_version,
        "output_sha256": sha256_file(args.output),
    }
    atomic_json(args.environment_output, environment)
    print(json.dumps(environment, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
