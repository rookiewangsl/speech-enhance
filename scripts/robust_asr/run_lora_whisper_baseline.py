#!/usr/bin/env python3
"""Evaluate one trained Whisper LoRA adapter with fixed WPE inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robust_asr.baseline import BaselineProgress, run_frozen_baseline
from robust_asr.config import load_json_object
from robust_asr.manifest import read_jsonl, write_jsonl_atomic
from robust_asr.models.whisper_lora import load_whisper_lora_for_inference
from robust_asr.paths import require_data_root
from robust_asr.training.evaluation import LoadedWhisperTranscriber


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--seed-predictions", type=Path)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--limit", type=int, default=1_000)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--frontends",
        nargs="+",
        choices=("raw", "s_wpe_10", "s_wpe_40", "m_wpe_10"),
        default=("raw", "m_wpe_10"),
    )
    parser.add_argument(
        "--rt60", type=float, nargs="+", default=(0.2, 0.4, 0.6, 0.8, 1.0)
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--manifest-name",
        default="aishell1_dev_model.jsonl",
        help="JSONL name under manifests/aishell1.",
    )
    parser.add_argument(
        "--rir-split",
        choices=("dev", "test"),
        default="dev",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def seed_predictions(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    expected_model_revision: str,
) -> int:
    """Seed a new output with compatible clean/raw checkpoint predictions."""

    source = Path(source_path)
    destination = Path(destination_path)
    if destination.exists():
        return 0
    rows = read_jsonl(source)
    if not rows:
        raise ValueError("seed prediction file is empty")
    revisions = {str(row.get("model_revision")) for row in rows}
    if revisions != {expected_model_revision}:
        raise ValueError(
            "seed predictions disagree with the loaded LoRA adapter: "
            f"{sorted(revisions)} != {[expected_model_revision]}"
        )
    conditions = {str(row.get("frontend")) for row in rows}
    if not conditions <= {"clean", "raw"} or "raw" not in conditions:
        raise ValueError("seed predictions must contain only clean/raw conditions")
    write_jsonl_atomic(destination, rows)
    return len(rows)


def progress_printer(every: int):
    if every <= 0:
        raise ValueError("--progress-every must be positive")

    def report(value: BaselineProgress) -> None:
        if value.stage == "start":
            print(
                f"LoRA frontend eval: {value.completed}/{value.total} resumed | "
                f"{value.total - value.completed} remaining",
                file=sys.stderr,
                flush=True,
            )
            return
        if value.stage == "complete":
            print(
                f"LoRA frontend eval complete: {value.completed}/{value.total} | "
                f"generated={value.generated}",
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
        print(
            f"Progress {value.completed}/{value.total} "
            f"[{100 * value.completed / value.total:3.0f}%] | "
            f"{speed:.2f} item/s | ETA {eta / 60:.1f} min",
            file=sys.stderr,
            flush=True,
        )

    return report


def main() -> None:
    args = arguments()
    if Path(args.output_name).name != args.output_name:
        raise ValueError("--output-name must be a basename")
    root = require_data_root(args.data_root)
    whisper = load_json_object(Path("configs/robust_asr/whisper.json"))
    components = load_whisper_lora_for_inference(
        adapter_path=args.adapter_path,
        model_id=str(whisper["model_id"]),
        revision=str(whisper["revision"]),
        cache_dir=root / "cache" / "huggingface",
        device=args.device,
        local_files_only=args.local_files_only,
    )
    transcriber = LoadedWhisperTranscriber(
        processor=components.processor,
        model=components.model,
        model_id=str(whisper["model_id"]),
        base_revision=str(whisper["revision"]),
        adapter_sha256=components.adapter_sha256,
        device=args.device,
        num_beams=int(whisper["num_beams"]),
    )
    output_path = root / "outputs" / args.output_name
    seeded = 0
    if args.seed_predictions is not None:
        seeded = seed_predictions(
            args.seed_predictions,
            output_path,
            expected_model_revision=transcriber.model_revision,
        )
    summary = run_frozen_baseline(
        manifest_path=root / "manifests" / "aishell1" / args.manifest_name,
        corpus_root=root / "corpora" / "aishell1",
        rir_manifest_path=(
            root / "rir" / "pyroom_v1" / f"{args.rir_split}.jsonl"
        ),
        rir_root=root / "rir" / "pyroom_v1",
        output_path=output_path,
        transcriber=transcriber,
        limit=args.limit,
        frontends=tuple(args.frontends),
        rt60_seconds=tuple(args.rt60),
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
        checkpoint_every_results=args.checkpoint_every,
        progress_callback=progress_printer(args.progress_every),
    )
    print(
        json.dumps(
            {
                "adapter_path": str(args.adapter_path.resolve()),
                "adapter_sha256": components.adapter_sha256,
                "seeded_rows": seeded,
                "result_rows": summary["result_rows"],
                "resumed_rows": summary["resumed_rows"],
                "generated_rows": summary["generated_rows"],
                "conditions": summary["conditions"],
                "paired_deltas": summary["paired_deltas"],
                "output_path": str(output_path),
                "summary_path": str(output_path.with_suffix(".summary.json")),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
