"""Dev-only ablation for bounded RNNoise residual-mixing controllers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_rnnoise import enhance_in_chunks, project_path
from speech_frontend.audio import (
    AudioData,
    read_audio,
    validate_aligned_pair,
    write_audio,
)
from speech_frontend.metrics import si_sdr, stoi
from speech_frontend.rnnoise import RNNoiseLibrary
from speech_frontend.rnnoise.controller import (
    CorrectionAwareConfig,
    correction_aware_mix,
    fixed_residual_mix,
    vad_aware_mix,
)

METHODS = (
    "r3_official",
    "r4a_fixed_050",
    "r4b_vad_protect_025",
    "r4c_correction_-11db",
    "r4c_correction_-9db",
    "r4c_correction_-7db",
    "r4d_correction_-9db_vad015",
    "r4d_correction_-9db_vad030",
)


def variants(
    noisy: np.ndarray,
    enhanced: np.ndarray,
    vad_probability: np.ndarray,
) -> dict[str, tuple[np.ndarray, float]]:
    output: dict[str, tuple[np.ndarray, float]] = {
        "r3_official": (enhanced, 1.0),
        "r4a_fixed_050": (
            fixed_residual_mix(noisy, enhanced, 0.50),
            0.50,
        ),
        "r4b_vad_protect_025": (
            vad_aware_mix(
                noisy,
                enhanced,
                vad_probability,
                speech_protection=0.25,
            ),
            float(1.0 - 0.25 * np.mean(vad_probability)),
        ),
    }
    for threshold in (-11.0, -9.0, -7.0):
        result = correction_aware_mix(
            noisy,
            enhanced,
            config=CorrectionAwareConfig(
                correction_threshold_db=threshold,
                speech_protection=0.0,
            ),
        )
        output[f"r4c_correction_{threshold:g}db"] = (
            result.samples,
            float(np.mean(result.frame_strength)),
        )
    protected = correction_aware_mix(
        noisy,
        enhanced,
        vad_probabilities=vad_probability,
        config=CorrectionAwareConfig(
            correction_threshold_db=-9.0,
            speech_protection=0.15,
        ),
    )
    output["r4d_correction_-9db_vad015"] = (
        protected.samples,
        float(np.mean(protected.frame_strength)),
    )
    stronger_protection = correction_aware_mix(
        noisy,
        enhanced,
        vad_probabilities=vad_probability,
        config=CorrectionAwareConfig(
            correction_threshold_db=-9.0,
            speech_protection=0.30,
        ),
    )
    output["r4d_correction_-9db_vad030"] = (
        stronger_protection.samples,
        float(np.mean(stronger_protection.frame_strength)),
    )
    return output


def summarize(
    rows: list[dict[str, str | float | int]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        si_sdri = np.asarray(
            [float(row["si_sdri_db"]) for row in method_rows]
        )
        stoi_change = np.asarray(
            [float(row["stoi_improvement"]) for row in method_rows]
        )
        summary[method] = {
            "files": len(method_rows),
            "mean_si_sdri_db": float(np.mean(si_sdri)),
            "median_si_sdri_db": float(np.median(si_sdri)),
            "positive_si_sdri_fraction": float(np.mean(si_sdri > 0)),
            "mean_stoi_improvement": float(np.mean(stoi_change)),
            "median_stoi_improvement": float(np.median(stoi_change)),
            "nonnegative_stoi_fraction": float(
                np.mean(stoi_change >= 0)
            ),
            "mean_strength": float(
                np.mean(
                    [float(row["mean_strength"]) for row in method_rows]
                )
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--chunk-size", type=int, default=137)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-audio", action="store_true")
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
    manifest_path = project_path(project_root, arguments.manifest)
    manifest_rows = [
        json.loads(line) for line in manifest_path.read_text().splitlines()
    ]
    if arguments.limit is not None:
        manifest_rows = manifest_rows[: arguments.limit]
    output_root = project_path(project_root, arguments.output_root)
    library = RNNoiseLibrary(
        project_path(project_root, arguments.library)
        if arguments.library is not None
        else None
    )
    rows: list[dict[str, str | float | int]] = []

    for manifest_row in manifest_rows:
        clean = read_audio(project_path(project_root, manifest_row["clean"]))
        noisy = read_audio(project_path(project_root, manifest_row["noisy"]))
        validate_aligned_pair(clean, noisy)
        input_si_sdr = si_sdr(clean.samples, noisy.samples)
        input_stoi = stoi(
            clean.samples,
            noisy.samples,
            sample_rate=clean.sample_rate,
        )
        enhanced, vad_probability, _ = enhance_in_chunks(
            noisy.samples,
            library,
            arguments.chunk_size,
        )
        file_variants = variants(
            noisy.samples,
            enhanced,
            vad_probability,
        )
        for method in arguments.methods:
            output, mean_strength = file_variants[method]
            output_si_sdr = si_sdr(clean.samples, output)
            output_stoi = stoi(
                clean.samples,
                output,
                sample_rate=clean.sample_rate,
            )
            rows.append(
                {
                    "file_id": manifest_row["id"],
                    "split": manifest_row.get("split", "unknown"),
                    "method": method,
                    "input_si_sdr_db": input_si_sdr,
                    "output_si_sdr_db": output_si_sdr,
                    "si_sdri_db": output_si_sdr - input_si_sdr,
                    "input_stoi": input_stoi,
                    "output_stoi": output_stoi,
                    "stoi_improvement": output_stoi - input_stoi,
                    "mean_strength": mean_strength,
                    "max_abs_output": float(
                        np.max(np.abs(output), initial=0.0)
                    ),
                    "clipping_samples": int(
                        np.count_nonzero(np.abs(output) > 1.0)
                    ),
                }
            )
            if arguments.save_audio:
                write_audio(
                    output_root / "audio" / method / f"{manifest_row['id']}.wav",
                    AudioData(output.astype(np.float32), noisy.sample_rate),
                )
        print(f"evaluated {manifest_row['id']}", flush=True)

    metrics_path = output_root / "metrics" / "controller_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result_summary = summarize(rows)
    summary_path = output_root / "metrics" / "controller_summary.json"
    summary_path.write_text(
        json.dumps(result_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result_summary, indent=2))


if __name__ == "__main__":
    main()
