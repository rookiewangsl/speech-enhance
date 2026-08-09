"""Small score-threshold sweep for Sohn-style statistical VAD."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from speech_frontend.audio import read_audio
from speech_frontend.vad import (
    StatisticalVAD,
    StatisticalVADConfig,
    binary_metrics,
    labels_from_intervals,
)
from speech_frontend.vad.state_machine import VADStateConfig, VADStateMachine


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--threshold-on", nargs="+", type=float, default=[0.02, 0.05, 0.10, 0.20, 0.40, 0.80])
    parser.add_argument("--threshold-gap", nargs="+", type=float, default=[0.01, 0.03, 0.08, 0.15])
    parser.add_argument("--alpha-dd", nargs="+", type=float, default=[0.92, 0.96, 0.98])
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/statistical_vad_tuning.json"))
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()
    rows = [
        json.loads(line)
        for line in resolve(project_root, str(arguments.manifest)).read_text().splitlines()
    ]
    mixtures = [
        (row, read_audio(resolve(project_root, row["noisy"])).samples)
        for row in rows
    ]
    results: list[dict[str, float]] = []
    for alpha_dd in arguments.alpha_dd:
        config = StatisticalVADConfig(alpha_dd=alpha_dd)
        detector = StatisticalVAD(config)
        cached_scores = []
        for row, noisy in mixtures:
            score = detector.detect(noisy).energy_db
            target = labels_from_intervals(
                score.size,
                frame_length=config.frame_length,
                hop_length=config.hop_length,
                intervals=tuple(tuple(interval) for interval in row["speech_intervals"]),
            )
            cached_scores.append((score, target))
        hop_ms = 1_000.0 * config.hop_length / config.sample_rate
        for threshold_on, gap in itertools.product(
            arguments.threshold_on,
            arguments.threshold_gap,
        ):
            threshold_off = threshold_on - gap
            if threshold_off < 0.0:
                continue
            state = VADStateMachine(
                VADStateConfig(
                    threshold_on=threshold_on,
                    threshold_off=threshold_off,
                    onset_frames=config.onset_frames,
                    hangover_frames=round(config.hangover_ms / hop_ms),
                    pre_roll_frames=round(config.pre_roll_ms / hop_ms),
                    minimum_speech_frames=round(config.minimum_speech_ms / hop_ms),
                )
            )
            scores = [
                binary_metrics(target, state.apply(score))
                for score, target in cached_scores
            ]
            results.append(
                {
                    "score_threshold_on": threshold_on,
                    "score_threshold_off": threshold_off,
                    "alpha_dd": alpha_dd,
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
