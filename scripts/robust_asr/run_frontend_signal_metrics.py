#!/usr/bin/env python3
"""Run parallel direct-target SI-SDR/STOI analysis for frozen WPE branches."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from robust_asr.frontend_metrics import (
    FrontendMetricProgress,
    run_frontend_signal_metrics,
)
from robust_asr.paths import require_data_root


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument(
        "--frontends",
        nargs="+",
        choices=("raw", "s_wpe_10", "s_wpe_40", "m_wpe_10"),
        default=("raw", "s_wpe_10", "s_wpe_40", "m_wpe_10"),
    )
    parser.add_argument(
        "--rt60", type=float, nargs="+", default=(0.2, 0.4, 0.6, 0.8, 1.0)
    )
    parser.add_argument(
        "--output-name",
        default="frontend_signal_metrics_dev_500utt_v1.jsonl",
    )
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def progress_printer(every: int):
    if every <= 0:
        raise ValueError("--progress-every must be positive")

    def report(value: FrontendMetricProgress) -> None:
        if (
            value.completed_jobs != 0
            and value.completed_jobs != value.total_jobs
            and value.completed_jobs % every != 0
        ):
            return
        speed = value.completed_jobs / max(value.elapsed_seconds, 1e-9)
        eta_text = (
            "pending"
            if value.completed_jobs == 0
            else f"{(value.total_jobs - value.completed_jobs) / speed / 60:.1f} min"
        )
        print(
            f"Signal metrics {value.completed_jobs}/{value.total_jobs} "
            f"[{100 * value.completed_jobs / value.total_jobs:3.0f}%] "
            f"| {speed:.2f} job/s | ETA {eta_text}",
            file=sys.stderr,
            flush=True,
        )

    return report


def main() -> None:
    args = arguments()
    if Path(args.output_name).name != args.output_name:
        raise ValueError("--output-name must be a basename")
    root = require_data_root(args.data_root)
    summary = run_frontend_signal_metrics(
        manifest_path=(
            root / "manifests" / "aishell1" / "aishell1_dev_frontend.jsonl"
        ),
        corpus_root=root / "corpora" / "aishell1",
        rir_manifest_path=root / "rir" / "pyroom_v1" / "dev.jsonl",
        rir_root=root / "rir" / "pyroom_v1",
        rir_validation_path=root / "rir" / "pyroom_v1" / "validation.json",
        output_path=root / "outputs" / args.output_name,
        limit=args.limit,
        frontends=tuple(args.frontends),
        rt60_seconds=tuple(args.rt60),
        workers=args.workers,
        progress_callback=progress_printer(args.progress_every),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
