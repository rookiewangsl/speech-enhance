"""Export reproducible four-condition audio inputs for ASR evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from speech_frontend.audio import (
    AudioData,
    read_audio,
    validate_aligned_pair,
    write_audio,
)
from speech_frontend.enhancement.wiener import WienerConfig
from speech_frontend.noise.mcra import MCRAConfig
from speech_frontend.pipeline import ClassicalEnhancer
from speech_frontend.rnnoise import RNNoiseLibrary, StreamingRNNoise16k
from speech_frontend.rnnoise.backend import default_library_path
from speech_frontend.stft import STFTConfig


CONDITIONS = ("clean", "noisy", "mcra_dd_wiener", "rnnoise_r3")
SCHEMA_VERSION = 1
FROZEN_RNNOISE_COMMIT = "70f1d256acd4b34a572f999a05c87bf00b67730d"
FROZEN_RNNOISE_ARCHIVE_SHA256 = (
    "0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37"
)
RNNoiseProcessor = Callable[
    [AudioData, Path | None, int],
    tuple[AudioData, dict[str, Any]],
]


def _implementation_provenance() -> dict[str, Any]:
    """Identify the local code that can change exported audio samples."""

    project_root = Path(__file__).resolve().parents[1]
    relative_sources = (
        "scripts/export_asr_inputs.py",
        "src/speech_frontend/pipeline.py",
        "src/speech_frontend/stft.py",
        "src/speech_frontend/noise/mcra.py",
        "src/speech_frontend/enhancement/wiener.py",
        "src/speech_frontend/rnnoise/backend.py",
        "src/speech_frontend/rnnoise/streaming.py",
    )
    source_sha256 = {
        relative: _sha256(project_root / relative) for relative in relative_sources
    }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision = "unavailable"
    return {
        "git_revision": revision,
        "source_sha256": source_sha256,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(config: dict[str, Any]) -> str:
    """Hash a config using a stable, whitespace-free JSON representation."""

    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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


def _atomic_write_audio(path: Path, audio: AudioData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".wav",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_audio(temporary, audio)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _delay_compensate(samples: np.ndarray, delay_samples: int) -> np.ndarray:
    if delay_samples < 0:
        raise ValueError("RNNoise alignment delay cannot be negative")
    if delay_samples == 0:
        return samples.copy()
    if delay_samples >= samples.size:
        return np.zeros_like(samples)
    return np.pad(samples[delay_samples:], (0, delay_samples))


def _run_rnnoise(
    audio: AudioData,
    library_path: Path | None,
    chunk_size: int,
) -> tuple[AudioData, dict[str, Any]]:
    """Run official R3 and return its delayed output plus timing metadata."""

    if audio.sample_rate != 16_000:
        raise ValueError("RNNoise R3 ASR export requires 16 kHz input")
    library = RNNoiseLibrary(library_path)
    stream = StreamingRNNoise16k(library)
    outputs: list[np.ndarray] = []
    started = time.perf_counter()
    for offset in range(0, audio.samples.size, chunk_size):
        result = stream.process_chunk(audio.samples[offset : offset + chunk_size])
        outputs.append(result.samples)
    result = stream.flush()
    outputs.append(result.samples)
    processing_seconds = time.perf_counter() - started
    enhanced = np.concatenate(outputs)
    return AudioData(enhanced.astype(np.float32), audio.sample_rate), {
        "processing_seconds": processing_seconds,
        "algorithmic_delay_samples": stream.algorithmic_delay_samples,
        "alignment_delay_samples": stream.alignment_delay_samples,
        "resampler_input_clipping_samples": stream.resampler_clipping_samples,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _build_condition_configs(
    protocol: dict[str, Any],
    rnnoise_provenance: dict[str, Any],
    *,
    library_path: Path | None,
    implementation_provenance: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], ClassicalEnhancer]:
    implementation_provenance = (
        implementation_provenance or _implementation_provenance()
    )
    selected = protocol.get("classical_enhancement", {})
    if selected.get("method") != "mcra_dd_wiener":
        raise ValueError("frozen protocol classical method must be mcra_dd_wiener")
    if float(selected.get("alpha_dd", -1)) != 0.92:
        raise ValueError("frozen MCRA + DD-Wiener alpha_dd must be 0.92")
    if float(selected.get("gain_floor", -1)) != 0.20:
        raise ValueError("frozen MCRA + DD-Wiener gain_floor must be 0.20")
    wiener = WienerConfig(
        alpha_dd=float(selected["alpha_dd"]),
        gain_floor=float(selected["gain_floor"]),
    )
    stft = STFTConfig()
    mcra = MCRAConfig()
    enhancer = ClassicalEnhancer(
        stft_config=stft,
        mcra_config=mcra,
        wiener_config=wiener,
    )
    chunk_size = int(protocol.get("neural_enhancement", {}).get("chunk_size_16k"))
    if chunk_size <= 0:
        raise ValueError("frozen RNNoise chunk_size_16k must be positive")
    if chunk_size != 137:
        raise ValueError("frozen RNNoise chunk_size_16k must be 137")
    if rnnoise_provenance.get("source", {}).get("commit") != FROZEN_RNNOISE_COMMIT:
        raise ValueError("RNNoise commit does not match frozen R3 protocol")
    if (
        rnnoise_provenance.get("model", {}).get("archive_sha256")
        != FROZEN_RNNOISE_ARCHIVE_SHA256
    ):
        raise ValueError("RNNoise model does not match frozen R3 protocol")
    rnnoise_config: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "condition": "rnnoise_r3",
        "implementation": "official_rnnoise_c_api_streaming_16k",
        "chunk_size": chunk_size,
        "delay_compensation": True,
        "pcm_compatible": False,
        "provenance": rnnoise_provenance,
        "implementation_provenance": implementation_provenance,
    }
    if library_path is not None and library_path.is_file():
        rnnoise_config["library_sha256"] = _sha256(library_path)
    configs = {
        condition: {
            "schema_version": SCHEMA_VERSION,
            "condition": condition,
            "operation": "lossless_sample_copy_to_float_wav",
            "output_subtype": "FLOAT",
            "implementation_provenance": implementation_provenance,
        }
        for condition in ("clean", "noisy")
    }
    configs["mcra_dd_wiener"] = {
        "schema_version": SCHEMA_VERSION,
        "condition": "mcra_dd_wiener",
        "implementation": "speech_frontend.pipeline.ClassicalEnhancer",
        "method": "mcra_dd_wiener",
        "stft": asdict(stft),
        "mcra": asdict(mcra),
        "wiener": asdict(wiener),
        "output_subtype": "FLOAT",
        "implementation_provenance": implementation_provenance,
    }
    configs["rnnoise_r3"] = rnnoise_config
    return configs, enhancer


def _cache_is_valid(
    audio_path: Path,
    sidecar_path: Path,
    *,
    file_id: str,
    condition: str,
    source_sha256: str,
    expected_config_digest: str,
    source_manifest_digest: str,
) -> dict[str, Any] | None:
    if not audio_path.is_file() or not sidecar_path.is_file():
        return None
    try:
        cached = _read_json(sidecar_path)
        if (
            cached.get("id") != file_id
            or cached.get("condition") != condition
            or cached.get("source_audio_sha256") != source_sha256
            or cached.get("enhancement_config_digest") != expected_config_digest
            or cached.get("source_manifest_digest") != source_manifest_digest
            or cached.get("audio_sha256") != _sha256(audio_path)
        ):
            return None
        audio = read_audio(audio_path)
        if (
            cached.get("sample_rate") != audio.sample_rate
            or cached.get("num_samples") != audio.samples.size
        ):
            return None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return cached


def _validate_manifest_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("manifest contains no rows")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each manifest row must be a JSON object")
        missing = {
            "id",
            "speaker_id",
            "split",
            "noise",
            "snr_db",
            "clean",
            "noisy",
            "sample_rate",
            "num_samples",
        } - row.keys()
        if missing:
            raise ValueError(f"manifest row missing fields: {sorted(missing)}")
        file_id = str(row["id"])
        if not file_id or file_id in {".", ".."} or Path(file_id).name != file_id:
            raise ValueError(f"unsafe utterance id: {file_id!r}")
        if file_id in seen:
            raise ValueError(f"duplicate utterance id: {file_id}")
        speaker_id = str(row["speaker_id"])
        if file_id.partition("_")[0] != speaker_id:
            raise ValueError(f"speaker_id does not match utterance id: {file_id}")
        for audio_field in ("clean", "noisy"):
            if Path(str(row[audio_field])).stem != file_id:
                raise ValueError(
                    f"{audio_field} filename does not match utterance id: {file_id}"
                )
        seen.add(file_id)


def export_asr_inputs(
    *,
    manifest_path: Path,
    project_root: Path,
    output_root: Path,
    output_manifest: Path,
    protocol_config_path: Path,
    rnnoise_config_path: Path,
    conditions: Iterable[str] = CONDITIONS,
    library_path: Path | None = None,
    limit: int | None = None,
    force: bool = False,
    rnnoise_processor: RNNoiseProcessor | None = None,
) -> list[dict[str, Any]]:
    """Export ASR inputs and return the output-manifest rows."""

    project_root = project_root.resolve()
    manifest_path = _resolve(project_root, manifest_path)
    output_root = _resolve(project_root, output_root)
    output_manifest = _resolve(project_root, output_manifest)
    protocol_config_path = _resolve(project_root, protocol_config_path)
    rnnoise_config_path = _resolve(project_root, rnnoise_config_path)
    chosen = tuple(conditions)
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("conditions must be non-empty and unique")
    unknown = set(chosen) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown ASR input conditions: {sorted(unknown)}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        rows = rows[:limit]
    _validate_manifest_rows(rows)

    resolved_library = None
    if "rnnoise_r3" in chosen:
        resolved_library = (
            _resolve(project_root, library_path)
            if library_path is not None
            else default_library_path().resolve()
        )
        if rnnoise_processor is None and not resolved_library.is_file():
            raise FileNotFoundError(
                f"RNNoise library not found: {resolved_library}. "
                "Run scripts/setup_rnnoise.sh first."
            )
    configs, enhancer = _build_condition_configs(
        _read_json(protocol_config_path),
        _read_json(rnnoise_config_path),
        library_path=resolved_library,
    )
    digests = {name: config_digest(value) for name, value in configs.items()}
    processor = rnnoise_processor or _run_rnnoise
    output_rows: list[dict[str, Any]] = []

    for source_row in rows:
        file_id = str(source_row["id"])
        source_manifest_digest = config_digest(source_row)
        clean_path = _resolve(project_root, source_row["clean"])
        noisy_path = _resolve(project_root, source_row["noisy"])
        clean = read_audio(clean_path)
        noisy = read_audio(noisy_path)
        validate_aligned_pair(clean, noisy)
        if clean.sample_rate != 16_000:
            raise ValueError(f"ASR export requires 16 kHz audio: {file_id}")
        if int(source_row["sample_rate"]) != clean.sample_rate:
            raise ValueError(f"manifest sample_rate mismatch: {file_id}")
        if int(source_row["num_samples"]) != clean.samples.size:
            raise ValueError(f"manifest num_samples mismatch: {file_id}")
        source_paths = {"clean": clean_path, "noisy": noisy_path}
        source_hashes = {name: _sha256(path) for name, path in source_paths.items()}

        for condition in chosen:
            source_kind = "clean" if condition == "clean" else "noisy"
            source_path = source_paths[source_kind]
            source_sha = source_hashes[source_kind]
            audio_path = output_root / "audio" / condition / f"{file_id}.wav"
            sidecar_path = audio_path.with_suffix(".json")
            if audio_path.resolve() in {clean_path, noisy_path}:
                raise ValueError("output path would overwrite source audio")
            cached = None if force else _cache_is_valid(
                audio_path,
                sidecar_path,
                file_id=file_id,
                condition=condition,
                source_sha256=source_sha,
                expected_config_digest=digests[condition],
                source_manifest_digest=source_manifest_digest,
            )
            if cached is not None:
                output_rows.append({**cached, "cache_status": "reused"})
                continue

            extra: dict[str, Any] = {}
            if condition == "clean":
                output = clean
                processing_seconds = 0.0
            elif condition == "noisy":
                output = noisy
                processing_seconds = 0.0
            elif condition == "mcra_dd_wiener":
                started = time.perf_counter()
                result = enhancer.enhance(noisy.samples, method="mcra_dd_wiener")
                processing_seconds = time.perf_counter() - started
                output = AudioData(result.samples.astype(np.float32), noisy.sample_rate)
            else:
                raw_output, rnnoise_metadata = processor(
                    noisy,
                    resolved_library,
                    int(configs[condition]["chunk_size"]),
                )
                if raw_output.sample_rate != noisy.sample_rate:
                    raise ValueError("RNNoise output sample rate changed")
                alignment_delay = int(rnnoise_metadata["alignment_delay_samples"])
                output = AudioData(
                    _delay_compensate(raw_output.samples, alignment_delay).astype(
                        np.float32
                    ),
                    raw_output.sample_rate,
                )
                processing_seconds = float(rnnoise_metadata["processing_seconds"])
                extra = {
                    **rnnoise_metadata,
                    "delay_compensated": True,
                    "alignment_delay_samples": alignment_delay,
                }
            if output.samples.shape != noisy.samples.shape:
                raise ValueError(f"{condition} output length changed for {file_id}")
            _atomic_write_audio(audio_path, output)
            generated = read_audio(audio_path)
            audio_sha = _sha256(audio_path)
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "id": file_id,
                "speaker_id": source_row.get("speaker_id", "unknown"),
                "split": source_row.get("split", "unknown"),
                "noise": source_row.get("noise", "unknown"),
                "snr_db": source_row.get("snr_db", "unknown"),
                "condition": condition,
                "audio": _portable_path(audio_path, project_root),
                "audio_sha256": audio_sha,
                "sample_rate": generated.sample_rate,
                "num_samples": int(generated.samples.size),
                "duration_seconds": generated.samples.size / generated.sample_rate,
                "source_audio": _portable_path(source_path, project_root),
                "source_audio_sha256": source_sha,
                "source_metadata": source_row,
                "source_manifest_digest": source_manifest_digest,
                "enhancement_config": configs[condition],
                "enhancement_config_digest": digests[condition],
                "processing_seconds": processing_seconds,
                "cache_status": "generated",
                **extra,
            }
            _atomic_write_text(
                sidecar_path,
                json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n",
            )
            output_rows.append(record)

    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in output_rows
    )
    _atomic_write_text(output_manifest, payload)
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=Path("configs/full_protocol.json"),
    )
    parser.add_argument(
        "--rnnoise-config",
        type=Path,
        default=Path("configs/rnnoise.json"),
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    project_root = arguments.project_root.resolve()
    output_root = _resolve(project_root, arguments.output_root)
    output_manifest = arguments.output_manifest or (
        output_root / "manifests" / f"{arguments.manifest.stem}_asr_inputs.jsonl"
    )
    rows = export_asr_inputs(
        manifest_path=arguments.manifest,
        project_root=project_root,
        output_root=output_root,
        output_manifest=output_manifest,
        protocol_config_path=arguments.protocol_config,
        rnnoise_config_path=arguments.rnnoise_config,
        conditions=arguments.conditions,
        library_path=arguments.library,
        limit=arguments.limit,
        force=arguments.force,
    )
    generated = sum(row["cache_status"] == "generated" for row in rows)
    print(
        json.dumps(
            {
                "manifest": str(_resolve(project_root, output_manifest)),
                "rows": len(rows),
                "generated": generated,
                "reused": len(rows) - generated,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
