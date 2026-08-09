"""Evaluate the Energy VAD against sample-exact synthetic labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from speech_frontend.audio import read_audio
from speech_frontend.vad import (
    EnergyVAD,
    EnergyVADConfig,
    StatisticalVAD,
    StatisticalVADConfig,
    WebRTCVAD,
    WebRTCVADConfig,
    binary_metrics,
    labels_from_intervals,
)


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/vad_metrics.csv"))
    parser.add_argument("--method", choices=("energy", "statistical", "webrtc"), default="energy")
    parser.add_argument("--threshold-on-db", type=float, default=12.0)
    parser.add_argument("--threshold-off-db", type=float, default=9.0)
    parser.add_argument("--noise-floor-rise-smoothing", type=float, default=0.98)
    parser.add_argument("--score-threshold-on", type=float, default=20.0)
    parser.add_argument("--score-threshold-off", type=float, default=10.0)
    parser.add_argument("--statistical-alpha-dd", type=float, default=0.96)
    parser.add_argument("--webrtc-aggressiveness", type=int, default=3)
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()
    rows = [
        json.loads(line)
        for line in resolve(project_root, str(arguments.manifest)).read_text().splitlines()
    ]
    if arguments.method == "energy":
        detector = EnergyVAD(
            EnergyVADConfig(
                threshold_on_db=arguments.threshold_on_db,
                threshold_off_db=arguments.threshold_off_db,
                noise_floor_rise_smoothing=arguments.noise_floor_rise_smoothing,
            )
        )
    elif arguments.method == "statistical":
        detector = StatisticalVAD(
            StatisticalVADConfig(
                score_threshold_on=arguments.score_threshold_on,
                score_threshold_off=arguments.score_threshold_off,
                alpha_dd=arguments.statistical_alpha_dd,
            )
        )
    else:
        detector = WebRTCVAD(
            WebRTCVADConfig(
                aggressiveness=arguments.webrtc_aggressiveness,
            )
        )
    if arguments.method == "webrtc":
        frame_length = (
            detector.config.frame_ms * detector.config.sample_rate // 1_000
        )
        hop_length = (
            detector.config.hop_ms * detector.config.sample_rate // 1_000
        )
    else:
        frame_length = detector.config.frame_length
        hop_length = detector.config.hop_length
    results: list[dict[str, float | int | str]] = []
    for row in rows:
        noisy = read_audio(resolve(project_root, row["noisy"]))
        result = detector.detect(noisy.samples)
        target = labels_from_intervals(
            result.frame_labels.size,
            frame_length=frame_length,
            hop_length=hop_length,
            intervals=tuple(tuple(interval) for interval in row["speech_intervals"]),
        )
        metrics = binary_metrics(target, result.frame_labels)
        results.append(
            {
                "file_id": row["id"],
                "method": arguments.method,
                "snr_db": row["snr_db"],
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "false_alarm_rate": metrics.false_alarm_rate,
                "miss_rate": metrics.miss_rate,
            }
        )
    output = resolve(project_root, str(arguments.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        metric: float(np.mean([row[metric] for row in results]))
        for metric in ("precision", "recall", "f1", "false_alarm_rate", "miss_rate")
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
