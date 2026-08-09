"""Run paired enhancement baselines and save per-file SI-SDR results."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import median

import numpy as np

from speech_frontend.audio import AudioData, read_audio, validate_aligned_pair, write_audio
from speech_frontend.enhancement.wiener import WienerConfig
from speech_frontend.metrics import si_sdr, stoi
from speech_frontend.pipeline import ClassicalEnhancer
from speech_frontend.vad.statistical import StatisticalVADConfig


METHODS = (
    "identity",
    "spectral_subtraction",
    "mcra_instantaneous_wiener",
    "mcra_dd_wiener",
    "mcra_om_lsa",
    "imcra_om_lsa",
    "dual_uncertainty_wiener",
)


def project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def read_manifest(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return rows if limit is None else rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument("--alpha-dd", type=float, default=0.96)
    parser.add_argument("--gain-floor", type=float, default=0.30)
    parser.add_argument("--gain-decrease-smoothing", type=float, default=0.7)
    parser.add_argument("--gain-increase-smoothing", type=float, default=0.4)
    parser.add_argument("--gain-frequency-smoothing", type=float, default=0.0)
    parser.add_argument("--startup-frames", type=int, default=0)
    parser.add_argument("--vad-prior-strength", type=float, default=1.0)
    parser.add_argument("--vad-score-threshold-on", type=float, default=10.0)
    parser.add_argument("--vad-score-threshold-off", type=float, default=9.5)
    parser.add_argument("--vad-alpha-dd", type=float, default=0.96)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit <= 0:
        raise ValueError("limit must be positive")

    project_root = arguments.project_root.resolve()
    rows = read_manifest(project_path(project_root, str(arguments.manifest)), arguments.limit)
    enhancer = ClassicalEnhancer(
        wiener_config=WienerConfig(
            alpha_dd=arguments.alpha_dd,
            gain_floor=arguments.gain_floor,
            gain_decrease_smoothing=arguments.gain_decrease_smoothing,
            gain_increase_smoothing=arguments.gain_increase_smoothing,
            gain_frequency_smoothing=arguments.gain_frequency_smoothing,
            startup_frames=arguments.startup_frames,
            vad_prior_strength=arguments.vad_prior_strength,
        ),
        statistical_vad_config=StatisticalVADConfig(
            score_threshold_on=arguments.vad_score_threshold_on,
            score_threshold_off=arguments.vad_score_threshold_off,
            alpha_dd=arguments.vad_alpha_dd,
        ),
    )
    results: list[dict[str, object]] = []

    for row in rows:
        clean = read_audio(project_path(project_root, row["clean"]))
        noisy = read_audio(project_path(project_root, row["noisy"]))
        validate_aligned_pair(clean, noisy)
        input_si_sdr = si_sdr(clean.samples, noisy.samples)
        input_stoi = stoi(
            clean.samples,
            noisy.samples,
            sample_rate=clean.sample_rate,
        )
        for method in arguments.methods:
            started = time.perf_counter()
            result = enhancer.enhance(noisy.samples, method=method)
            elapsed = time.perf_counter() - started
            output = AudioData(result.samples.astype(np.float32), noisy.sample_rate)
            if arguments.save_audio:
                output_path = (
                    arguments.output_root
                    / "audio"
                    / method
                    / f"{row['id']}.wav"
                )
                write_audio(output_path, output)
            output_si_sdr = si_sdr(clean.samples, output.samples)
            output_stoi = stoi(
                clean.samples,
                output.samples,
                sample_rate=clean.sample_rate,
            )
            results.append(
                {
                    "file_id": row["id"],
                    "speaker_id": row.get("speaker_id", "unknown"),
                    "method": method,
                    "split": row.get("split", "unknown"),
                    "noise": row.get("noise", "unknown"),
                    "snr_db": row.get("snr_db", "unknown"),
                    "alpha_dd": arguments.alpha_dd,
                    "gain_floor": arguments.gain_floor,
                    "gain_decrease_smoothing": arguments.gain_decrease_smoothing,
                    "gain_increase_smoothing": arguments.gain_increase_smoothing,
                    "gain_frequency_smoothing": arguments.gain_frequency_smoothing,
                    "startup_frames": arguments.startup_frames,
                    "input_si_sdr_db": input_si_sdr,
                    "output_si_sdr_db": output_si_sdr,
                    "si_sdri_db": output_si_sdr - input_si_sdr,
                    "input_stoi": input_stoi,
                    "output_stoi": output_stoi,
                    "stoi_improvement": output_stoi - input_stoi,
                    "rtf": elapsed / (clean.samples.size / clean.sample_rate),
                    "max_abs_output": float(np.max(np.abs(output.samples))),
                    "clipping_samples": int(np.count_nonzero(np.abs(output.samples) > 1.0)),
                }
            )
        print(f"evaluated {row['id']}")

    metrics_path = arguments.output_root / "metrics" / "enhancement_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    summary: dict[str, dict[str, float]] = {}
    for method in arguments.methods:
        method_rows = [row for row in results if row["method"] == method]
        summary[method] = {
            "files": len(method_rows),
            "mean_si_sdri_db": float(np.mean([row["si_sdri_db"] for row in method_rows])),
            "median_si_sdri_db": float(median(row["si_sdri_db"] for row in method_rows)),
            "mean_stoi_improvement": float(
                np.mean([row["stoi_improvement"] for row in method_rows])
            ),
            "mean_rtf": float(np.mean([row["rtf"] for row in method_rows])),
        }
    summary_path = arguments.output_root / "metrics" / "enhancement_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
