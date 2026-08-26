#!/usr/bin/env python3
"""Run a resumable clean/reverb/WPE Whisper-small baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_asr.baseline import run_frozen_baseline
from robust_asr.models.whisper_inference import FrozenWhisper
from robust_asr.paths import require_data_root


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--frontends",
        nargs="+",
        choices=("raw", "s_wpe_10", "s_wpe_40", "m_wpe_10"),
        default=("raw", "s_wpe_10", "s_wpe_40", "m_wpe_10"),
    )
    parser.add_argument(
        "--rt60", type=float, nargs="+", default=(0.2, 0.6, 1.0)
    )
    parser.add_argument("--rir-split", default="smoke")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    model = FrozenWhisper(
        cache_dir=root / "cache" / "huggingface",
        device=args.device,
        local_files_only=args.local_files_only,
    )
    summary = run_frozen_baseline(
        manifest_path=(
            root / "manifests" / "aishell1" / "aishell1_dev_frontend.jsonl"
        ),
        corpus_root=root / "corpora" / "aishell1",
        rir_manifest_path=root / "rir" / "pyroom_v1" / f"{args.rir_split}.jsonl",
        rir_root=root / "rir" / "pyroom_v1",
        output_path=root / "outputs" / "frozen_whisper_dev_smoke.jsonl",
        transcriber=model,
        limit=args.limit,
        frontends=tuple(args.frontends),
        rt60_seconds=tuple(args.rt60),
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
