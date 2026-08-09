"""Evaluate the R3 stateful RNNoise path on a paired manifest."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from speech_frontend.audio import (
    AudioData,
    read_audio,
    validate_aligned_pair,
    write_audio,
)
from speech_frontend.metrics import si_sdr, stoi
from speech_frontend.rnnoise import RNNoiseLibrary, StreamingRNNoise16k


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def compensate_delay(
    samples: np.ndarray,
    delay_samples: int,
) -> np.ndarray:
    if delay_samples >= samples.size:
        return np.zeros_like(samples)
    return np.pad(samples[delay_samples:], (0, delay_samples))


def enhance_in_chunks(
    samples: np.ndarray,
    library: RNNoiseLibrary,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    stream = StreamingRNNoise16k(library)
    output: list[np.ndarray] = []
    vad: list[np.ndarray] = []
    started = time.perf_counter()
    for offset in range(0, samples.size, chunk_size):
        result = stream.process_chunk(samples[offset : offset + chunk_size])
        output.append(result.samples)
        vad.append(result.vad_probabilities)
    result = stream.flush()
    output.append(result.samples)
    vad.append(result.vad_probabilities)
    elapsed = time.perf_counter() - started
    enhanced = np.concatenate(output)
    vad_probability = np.concatenate(vad)
    enhanced = compensate_delay(
        enhanced,
        stream.alignment_delay_samples,
    )
    metadata: dict[str, float | int] = {
        "algorithmic_delay_samples": stream.algorithmic_delay_samples,
        "algorithmic_delay_ms": (
            1_000 * stream.algorithmic_delay_samples / 16_000
        ),
        "file_alignment_delay_samples": stream.alignment_delay_samples,
        "file_alignment_delay_ms": (
            1_000 * stream.alignment_delay_samples / 16_000
        ),
        "processing_seconds": elapsed,
        "vad_frames": vad_probability.size,
        "vad_mean": float(np.mean(vad_probability)),
        "resampler_input_clipping_samples": (
            stream.resampler_clipping_samples
        ),
    }
    return enhanced, vad_probability, metadata


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int = 20260724,
    draws: int = 2_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sample = rng.choice(values, size=values.size, replace=True)
        means[index] = np.mean(sample)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, object]:
    si_sdri = np.asarray(
        [float(row["si_sdri_db"]) for row in rows],
        dtype=np.float64,
    )
    stoi_change = np.asarray(
        [float(row["stoi_improvement"]) for row in rows],
        dtype=np.float64,
    )
    rtf = np.asarray(
        [float(row["rtf_processing_only"]) for row in rows],
        dtype=np.float64,
    )
    ci_low, ci_high = bootstrap_mean_interval(si_sdri)
    return {
        "method": "R3_official_rnnoise_c_api_streaming",
        "files": len(rows),
        "mean_si_sdri_db": float(np.mean(si_sdri)),
        "median_si_sdri_db": float(np.median(si_sdri)),
        "si_sdri_p10_db": float(np.quantile(si_sdri, 0.10)),
        "si_sdri_p25_db": float(np.quantile(si_sdri, 0.25)),
        "si_sdri_p75_db": float(np.quantile(si_sdri, 0.75)),
        "si_sdri_p90_db": float(np.quantile(si_sdri, 0.90)),
        "mean_si_sdri_95ci_db": [ci_low, ci_high],
        "positive_si_sdri_fraction": float(np.mean(si_sdri > 0)),
        "mean_stoi_improvement": float(np.mean(stoi_change)),
        "median_stoi_improvement": float(np.median(stoi_change)),
        "nonnegative_stoi_fraction": float(np.mean(stoi_change >= 0)),
        "mean_rtf_processing_only": float(np.mean(rtf)),
        "median_rtf_processing_only": float(np.median(rtf)),
        "total_clipping_samples": int(
            sum(int(row["clipping_samples"]) for row in rows)
        ),
        "total_resampler_input_clipping_samples": int(
            sum(
                int(row["resampler_input_clipping_samples"])
                for row in rows
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--chunk-size", type=int, default=137)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-audio", action="store_true")
    arguments = parser.parse_args()
    if arguments.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if arguments.limit is not None and arguments.limit <= 0:
        raise ValueError("limit must be positive")

    project_root = arguments.project_root.resolve()
    manifest_path = project_path(project_root, arguments.manifest)
    manifest_rows = [
        json.loads(line) for line in manifest_path.read_text().splitlines()
    ]
    if arguments.limit is not None:
        manifest_rows = manifest_rows[: arguments.limit]
    if not manifest_rows:
        raise ValueError("manifest contains no rows")

    output_root = project_path(project_root, arguments.output_root)
    library_path = (
        project_path(project_root, arguments.library)
        if arguments.library is not None
        else None
    )
    library = RNNoiseLibrary(library_path)
    rows: list[dict[str, float | int | str]] = []

    for manifest_row in manifest_rows:
        clean = read_audio(project_path(project_root, manifest_row["clean"]))
        noisy = read_audio(project_path(project_root, manifest_row["noisy"]))
        validate_aligned_pair(clean, noisy)
        if noisy.sample_rate != 16_000:
            raise ValueError("R3 manifest evaluation currently requires 16 kHz")

        input_si_sdr = si_sdr(clean.samples, noisy.samples)
        input_stoi = stoi(
            clean.samples,
            noisy.samples,
            sample_rate=clean.sample_rate,
        )
        enhanced, _, metadata = enhance_in_chunks(
            noisy.samples,
            library,
            arguments.chunk_size,
        )
        output_si_sdr = si_sdr(clean.samples, enhanced)
        output_stoi = stoi(
            clean.samples,
            enhanced,
            sample_rate=clean.sample_rate,
        )
        duration = noisy.samples.size / noisy.sample_rate
        row: dict[str, float | int | str] = {
            "file_id": manifest_row["id"],
            "speaker_id": manifest_row.get("speaker_id", "unknown"),
            "split": manifest_row.get("split", "unknown"),
            "method": "R3_official_rnnoise_c_api_streaming",
            "chunk_size": arguments.chunk_size,
            "input_si_sdr_db": input_si_sdr,
            "output_si_sdr_db": output_si_sdr,
            "si_sdri_db": output_si_sdr - input_si_sdr,
            "input_stoi": input_stoi,
            "output_stoi": output_stoi,
            "stoi_improvement": output_stoi - input_stoi,
            "rtf_processing_only": (
                float(metadata["processing_seconds"]) / duration
            ),
            "algorithmic_delay_samples": int(
                metadata["algorithmic_delay_samples"]
            ),
            "algorithmic_delay_ms": float(
                metadata["algorithmic_delay_ms"]
            ),
            "file_alignment_delay_samples": int(
                metadata["file_alignment_delay_samples"]
            ),
            "file_alignment_delay_ms": float(
                metadata["file_alignment_delay_ms"]
            ),
            "vad_frames": int(metadata["vad_frames"]),
            "vad_mean": float(metadata["vad_mean"]),
            "max_abs_output": float(np.max(np.abs(enhanced), initial=0.0)),
            "clipping_samples": int(
                np.count_nonzero(np.abs(enhanced) > 1.0)
            ),
            "resampler_input_clipping_samples": int(
                metadata["resampler_input_clipping_samples"]
            ),
        }
        rows.append(row)
        if arguments.save_audio:
            write_audio(
                output_root / "audio" / "r3_rnnoise" / f"{manifest_row['id']}.wav",
                AudioData(enhanced.astype(np.float32), noisy.sample_rate),
            )
        print(
            f"{manifest_row['id']}: "
            f"SI-SDRi={row['si_sdri_db']:.3f} dB, "
            f"STOIΔ={row['stoi_improvement']:+.4f}, "
            f"RTF={row['rtf_processing_only']:.3f}",
            flush=True,
        )

    metrics_path = output_root / "metrics" / "rnnoise_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_path = output_root / "metrics" / "rnnoise_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
