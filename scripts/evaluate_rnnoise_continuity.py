"""Evaluate output-only RNNoise continuity stabilization."""

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
from speech_frontend.rnnoise import (
    ContinuityConfig,
    RNNoiseLibrary,
    stabilize_output_continuity,
)

METHODS = (
    "r3_official",
    "c1_continuity_3db",
    "c1_continuity_6db",
)


def continuity_variants(
    enhanced: np.ndarray,
    vad_probability: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "r3_official": (
            enhanced,
            np.zeros(
                int(np.ceil(enhanced.size / 160)),
                dtype=np.float32,
            ),
        )
    }
    for maximum in (3.0, 6.0):
        result = stabilize_output_continuity(
            enhanced,
            vad_probability,
            config=ContinuityConfig(
                allowed_drop_db=4.0,
                envelope_release_db_per_frame=1.5,
                max_boost_db=maximum,
            ),
        )
        variants[f"c1_continuity_{int(maximum)}db"] = (
            result.samples,
            result.frame_gain_db,
        )
    return variants


def summarize(
    rows: list[dict[str, str | float | int]],
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        si_sdri = np.asarray([row["si_sdri_db"] for row in selected])
        stoi_change = np.asarray(
            [row["stoi_improvement"] for row in selected]
        )
        output[method] = {
            "files": len(selected),
            "mean_si_sdri_db": float(np.mean(si_sdri)),
            "median_si_sdri_db": float(np.median(si_sdri)),
            "positive_si_sdri_fraction": float(np.mean(si_sdri > 0.0)),
            "mean_stoi_improvement": float(np.mean(stoi_change)),
            "median_stoi_improvement": float(np.median(stoi_change)),
            "nonnegative_stoi_fraction": float(
                np.mean(stoi_change >= 0.0)
            ),
            "mean_boosted_frame_fraction": float(
                np.mean([row["boosted_frame_fraction"] for row in selected])
            ),
            "mean_boost_db": float(
                np.mean([row["mean_boost_db"] for row in selected])
            ),
        }
    return output


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
        variants = continuity_variants(enhanced, vad_probability)
        for method, (output, frame_gain_db) in variants.items():
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
                    "boosted_frame_fraction": float(
                        np.mean(frame_gain_db > 1e-4)
                    ),
                    "mean_boost_db": float(np.mean(frame_gain_db)),
                    "max_boost_db": float(
                        np.max(frame_gain_db, initial=0.0)
                    ),
                    "max_abs_output": float(
                        np.max(np.abs(output), initial=0.0)
                    ),
                    "clipping_samples": int(
                        np.count_nonzero(np.abs(output) >= 1.0)
                    ),
                }
            )
            if arguments.save_audio:
                write_audio(
                    output_root
                    / "audio"
                    / method
                    / f"{manifest_row['id']}.wav",
                    AudioData(output.astype(np.float32), noisy.sample_rate),
                )
        print(f"evaluated {manifest_row['id']}", flush=True)

    metrics_path = output_root / "metrics" / "continuity_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    result_summary = summarize(rows)
    summary_path = output_root / "metrics" / "continuity_summary.json"
    summary_path.write_text(
        json.dumps(result_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result_summary, indent=2))


if __name__ == "__main__":
    main()
