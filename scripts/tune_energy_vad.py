"""Interpretably tune the Energy VAD on labeled synthetic dev mixtures."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from speech_frontend.audio import read_audio
from speech_frontend.vad import (
    EnergyVAD,
    EnergyVADConfig,
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
    parser.add_argument("--threshold-on", nargs="+", type=float, default=[8.0, 10.0, 12.0, 15.0])
    parser.add_argument("--threshold-gap", nargs="+", type=float, default=[3.0, 4.0, 6.0])
    parser.add_argument("--noise-floor-rise", nargs="+", type=float, default=[0.70, 0.80, 0.90, 0.95, 0.98])
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/energy_vad_tuning.json"))
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()
    rows = [
        json.loads(line)
        for line in resolve(project_root, str(arguments.manifest)).read_text().splitlines()
    ]
    mixtures = [
        (
            row,
            read_audio(resolve(project_root, row["noisy"])).samples,
        )
        for row in rows
    ]
    results: list[dict[str, float]] = []
    for threshold_on, gap, rise in itertools.product(
        arguments.threshold_on,
        arguments.threshold_gap,
        arguments.noise_floor_rise,
    ):
        config = EnergyVADConfig(
            threshold_on_db=threshold_on,
            threshold_off_db=threshold_on - gap,
            noise_floor_rise_smoothing=rise,
        )
        detector = EnergyVAD(config)
        scores = []
        for row, noisy in mixtures:
            prediction = detector.detect(noisy).frame_labels
            target = labels_from_intervals(
                prediction.size,
                frame_length=config.frame_length,
                hop_length=config.hop_length,
                intervals=tuple(tuple(interval) for interval in row["speech_intervals"]),
            )
            scores.append(binary_metrics(target, prediction))
        results.append(
            {
                "threshold_on_db": threshold_on,
                "threshold_off_db": threshold_on - gap,
                "noise_floor_rise_smoothing": rise,
                "mean_precision": float(np.mean([item.precision for item in scores])),
                "mean_recall": float(np.mean([item.recall for item in scores])),
                "mean_f1": float(np.mean([item.f1 for item in scores])),
                "mean_false_alarm_rate": float(np.mean([item.false_alarm_rate for item in scores])),
            }
        )
    results.sort(
        key=lambda result: (
            result["mean_f1"],
            result["mean_precision"],
            result["mean_recall"],
        ),
        reverse=True,
    )
    output = resolve(project_root, str(arguments.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results[:10], indent=2))


if __name__ == "__main__":
    main()
