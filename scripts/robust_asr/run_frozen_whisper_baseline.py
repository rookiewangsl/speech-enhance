#!/usr/bin/env python3
"""Run a resumable clean/reverb/WPE Whisper-small baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robust_asr.baseline import BaselineProgress, run_frozen_baseline
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
    parser.add_argument(
        "--manifest-name",
        default="aishell1_dev_frontend.jsonl",
        help="JSONL name under manifests/aishell1.",
    )
    parser.add_argument(
        "--output-name",
        help="JSONL basename under outputs; defaults to a run-specific name.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    return parser.parse_args()


def _duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, final_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{final_seconds:02d}"
    return f"{minutes:02d}:{final_seconds:02d}"


def progress_printer(*, every: int, quiet: bool):
    if every <= 0:
        raise ValueError("--progress-every must be positive")
    live = sys.stderr.isatty()
    active = False

    def finish_line() -> None:
        nonlocal active
        if active:
            sys.stderr.write("\n")
            sys.stderr.flush()
            active = False

    def report(value: BaselineProgress) -> None:
        nonlocal active
        if quiet:
            return
        if value.stage == "start":
            print(
                f"Baseline: {value.completed}/{value.total} resumed "
                f"| {value.total - value.completed} remaining",
                file=sys.stderr,
                flush=True,
            )
            return
        if value.stage == "complete":
            finish_line()
            print(
                f"Baseline complete: {value.completed}/{value.total} "
                f"| generated={value.generated} "
                f"| time={_duration(value.elapsed_seconds)}",
                file=sys.stderr,
                flush=True,
            )
            return
        if (
            value.generated != 1
            and value.completed != value.total
            and value.generated % every != 0
        ):
            return
        speed = value.generated / max(value.elapsed_seconds, 1e-9)
        eta = (value.total - value.completed) / max(speed, 1e-9)
        percent = 100.0 * value.completed / value.total
        condition = value.frontend or "unknown"
        if value.rt60_seconds is not None:
            condition += f"@{value.rt60_seconds:.1f}s"
        line = (
            f"Progress {value.completed}/{value.total} [{percent:3.0f}%] "
            f"| {condition} | {speed:.2f} item/s | ETA {_duration(eta)}"
        )
        if live:
            sys.stderr.write("\r\033[2K" + line)
            sys.stderr.flush()
            active = True
        else:
            print(line, file=sys.stderr, flush=True)

    return report


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    manifest_path = root / "manifests" / "aishell1" / args.manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if args.output_name is not None and Path(args.output_name).name != args.output_name:
        raise ValueError("--output-name must be a basename, not a path")
    output_name = args.output_name or (
        f"frozen_whisper_{Path(args.manifest_name).stem}_"
        f"{args.rir_split}_{args.limit}utt.jsonl"
    )
    if not args.quiet:
        print("Loading frozen Whisper model...", file=sys.stderr, flush=True)
    model = FrozenWhisper(
        cache_dir=root / "cache" / "huggingface",
        device=args.device,
        local_files_only=args.local_files_only,
    )
    summary = run_frozen_baseline(
        manifest_path=manifest_path,
        corpus_root=root / "corpora" / "aishell1",
        rir_manifest_path=root / "rir" / "pyroom_v1" / f"{args.rir_split}.jsonl",
        rir_root=root / "rir" / "pyroom_v1",
        output_path=root / "outputs" / output_name,
        transcriber=model,
        limit=args.limit,
        frontends=tuple(args.frontends),
        rt60_seconds=tuple(args.rt60),
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
        checkpoint_every_results=args.checkpoint_every,
        progress_callback=progress_printer(
            every=args.progress_every,
            quiet=args.quiet,
        ),
    )
    if args.print_summary:
        payload = summary
    else:
        payload = {
            "run_protocol_sha256": summary["run_protocol_sha256"],
            "result_rows": summary["result_rows"],
            "resumed_rows": summary["resumed_rows"],
            "generated_rows": summary["generated_rows"],
            "conditions": summary["conditions"],
            "raw_robustness": summary["raw_robustness"],
            "output_path": str(root / "outputs" / output_name),
            "summary_path": str(
                (root / "outputs" / output_name).with_suffix(".summary.json")
            ),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
