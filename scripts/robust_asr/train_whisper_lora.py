#!/usr/bin/env python3
"""Train one frozen-protocol Whisper LoRA adapter on clean or MCT data."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from robust_asr.config import canonical_sha256, load_json_object
from robust_asr.lora import LoRAProtocol, LoRATarget
from robust_asr.manifest import read_jsonl
from robust_asr.models.whisper_lora import load_whisper_lora_components
from robust_asr.paths import require_data_root
from robust_asr.protocol import load_and_validate_protocol
from robust_asr.training.data import WhisperAdaptationDataset, WhisperBatchCollator
from robust_asr.training.engine import TrainingConfig, run_lora_training
from robust_asr.training.evaluation import WhisperDevEvaluator
from robust_asr.training.reporting import (
    ConsoleTrainingReporter,
    RunOverview,
    StructuredTrainingLogger,
    TrainingReporter,
)
from robust_asr.training.selection import CheckpointSelector


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--mode", choices=("clean", "mct"), required=True)
    parser.add_argument(
        "--lora-target",
        choices=tuple(value.value for value in LoRATarget),
        default=LoRATarget.ENCODER_QV.value,
    )
    parser.add_argument("--train-hours", type=int, choices=(5, 10, 20), default=20)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--maximum-optimizer-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--console-every", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        help="W0 dev_model summary; defaults under the data-root outputs directory.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _versions(names: tuple[str, ...]) -> dict[str, str]:
    output: dict[str, str] = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "missing"
    return output


def _git_revision() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40:
        raise ValueError("Git HEAD is not a full 40-character revision")
    return revision


def load_w0_dev_baseline(
    path: str | Path,
    *,
    model_id: str,
    model_revision: str,
    expected_utterances: int,
    expected_rt60: tuple[float, ...],
) -> tuple[float, dict[str, Any]]:
    source = Path(path)
    summary = load_json_object(source)
    if summary.get("model_id") != model_id:
        raise ValueError("W0 baseline model id disagrees with the training protocol")
    if summary.get("model_revision") != model_revision:
        raise ValueError("W0 baseline revision disagrees with the training protocol")
    if int(summary.get("utterance_limit", -1)) != expected_utterances:
        raise ValueError("W0 baseline does not use the frozen dev_model size")
    if tuple(map(float, summary.get("rt60_seconds", ()))) != expected_rt60:
        raise ValueError("W0 baseline RT60 grid disagrees with the training protocol")
    if set(summary.get("frontends", ())) != {"raw"}:
        raise ValueError("W0 baseline must contain the raw-only dev selection matrix")
    clean = [
        row
        for row in summary.get("conditions", ())
        if row.get("frontend") == "clean"
        and row.get("target_rt60_seconds") is None
    ]
    if len(clean) != 1:
        raise ValueError("W0 baseline has no unique clean condition")
    clean_cer = float(clean[0]["cer"])
    if not 0 <= clean_cer <= 1:
        raise ValueError("W0 clean CER must be a fraction in [0, 1]")
    return clean_cer, summary


def main() -> None:
    args = arguments()
    if Path(args.experiment_id).name != args.experiment_id:
        raise ValueError("--experiment-id must be a basename")
    root = require_data_root(args.data_root)
    config_root = Path("configs/robust_asr")
    protocol = load_and_validate_protocol(config_root)
    data_config = load_json_object(config_root / "data.json")
    rir_config = load_json_object(config_root / "rir.json")
    lora_config = load_json_object(config_root / "lora.json")
    whisper_config = load_json_object(config_root / "whisper.json")
    evaluation_config = load_json_object(config_root / "evaluation.json")

    output_dir = root / "runs" / args.experiment_id
    if (output_dir / "training_summary.json").exists():
        raise FileExistsError(f"training run is already complete: {output_dir}")
    train_manifest = (
        root
        / "manifests"
        / "aishell1"
        / f"aishell1_train_{args.train_hours}h.jsonl"
    )
    dev_manifest = root / "manifests" / "aishell1" / "aishell1_dev_model.jsonl"
    baseline_path = args.baseline_summary or (
        root
        / "outputs"
        / "frozen_whisper_dev_model_dev_1000utt_v1.summary.json"
    )
    rt60_grid = tuple(map(float, evaluation_config["rt60_seconds"]))
    baseline_clean_cer, baseline_summary = load_w0_dev_baseline(
        baseline_path,
        model_id=str(whisper_config["model_id"]),
        model_revision=str(whisper_config["revision"]),
        expected_utterances=int(data_config["dev_model_utterances"]),
        expected_rt60=rt60_grid,
    )
    if args.mode == "mct":
        validation = load_json_object(
            root / "rir" / "pyroom_v1" / "validation.json"
        )
        if validation.get("status") != "PASS" or not validation.get("verify_files"):
            raise ValueError("MCT training requires a passed full-file RIR validation")

    epochs = int(lora_config["max_epochs"] if args.epochs is None else args.epochs)
    batch_size = int(
        lora_config["per_device_train_batch_size"]
        if args.batch_size is None
        else args.batch_size
    )
    accumulation = int(
        lora_config["gradient_accumulation_steps"]
        if args.gradient_accumulation is None
        else args.gradient_accumulation
    )
    training = TrainingConfig(
        epochs=epochs,
        per_device_batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        learning_rate=float(lora_config["learning_rate"]),
        weight_decay=float(lora_config["weight_decay"]),
        warmup_ratio=float(lora_config["warmup_ratio"]),
        max_grad_norm=float(lora_config["max_grad_norm"]),
        seed=int(lora_config["seed"]),
        num_workers=int(
            lora_config["dataloader_num_workers"]
            if args.num_workers is None
            else args.num_workers
        ),
        prefetch_factor=int(
            lora_config["dataloader_prefetch_factor"]
            if args.prefetch_factor is None
            else args.prefetch_factor
        ),
        precision=str(lora_config["precision"]),
        maximum_optimizer_steps=args.maximum_optimizer_steps,
    )
    target = LoRATarget(args.lora_target)
    dataset = WhisperAdaptationDataset(
        manifest_path=train_manifest,
        corpus_root=root / "corpora" / "aishell1",
        mode=args.mode,
        rir_manifest_path=(
            None
            if args.mode == "clean"
            else root / "rir" / "pyroom_v1" / "train.jsonl"
        ),
        rir_root=(
            None if args.mode == "clean" else root / "rir" / "pyroom_v1"
        ),
        reverb_probability=(
            0.0
            if args.mode == "clean"
            else float(data_config["mct_raw_reverb_probability"])
        ),
        seed=training.seed,
        reference_channel=int(rir_config["reference_channel"]),
        target_rms_dbfs=float(rir_config["target_rms_dbfs"]),
        peak_headroom_db=float(rir_config["peak_headroom_db"]),
    )
    components = load_whisper_lora_components(
        model_id=str(whisper_config["model_id"]),
        target=target,
        protocol=LoRAProtocol(
            rank=int(lora_config["rank"]),
            alpha=int(lora_config["alpha"]),
            dropout=float(lora_config["dropout"]),
            bias=str(lora_config["bias"]),
        ),
        revision=str(whisper_config["revision"]),
        cache_dir=root / "cache" / "huggingface",
        local_files_only=args.local_files_only,
    )
    collator = WhisperBatchCollator(
        components.processor,
        decoder_start_token_id=components.model.config.decoder_start_token_id,
    )
    evaluator = WhisperDevEvaluator(
        processor=components.processor,
        manifest_path=dev_manifest,
        corpus_root=root / "corpora" / "aishell1",
        rir_manifest_path=root / "rir" / "pyroom_v1" / "dev.jsonl",
        rir_root=root / "rir" / "pyroom_v1",
        output_dir=output_dir,
        model_id=str(whisper_config["model_id"]),
        base_revision=str(whisper_config["revision"]),
        limit=int(data_config["dev_model_utterances"]),
        rt60_seconds=rt60_grid,
        heavy_rt60_seconds=tuple(
            map(float, lora_config["logging"]["heavy_rt60_seconds"])
        ),
        device="cuda",
        num_beams=int(whisper_config["num_beams"]),
        seed=training.seed,
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal Whisper LoRA training requires CUDA")
    device_properties = torch.cuda.get_device_properties(0)
    train_rows = read_jsonl(train_manifest)
    dev_rows = read_jsonl(dev_manifest)
    train_hours = sum(float(row["duration_seconds"]) for row in train_rows) / 3600
    trainable_ratio = components.trainable_parameters / components.total_parameters
    overview = RunOverview(
        experiment_id=args.experiment_id,
        model_name=str(whisper_config["model_id"]),
        lora_target=target.value,
        lora_rank=int(lora_config["rank"]),
        trainable_ratio=trainable_ratio,
        train_hours=train_hours,
        train_utterances=len(dataset),
        dev_utterances=len(dev_rows),
        clean_probability=(
            1.0
            if args.mode == "clean"
            else float(data_config["mct_clean_probability"])
        ),
        reverb_probability=(
            0.0
            if args.mode == "clean"
            else float(data_config["mct_raw_reverb_probability"])
        ),
        precision=training.precision,
        per_device_batch_size=training.per_device_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        epochs=training.epochs,
        device_name=device_properties.name,
        device_memory_gib=device_properties.total_memory / 1024**3,
        output_dir=output_dir,
    )
    run_config = {
        "schema_version": 1,
        "protocol_sha256": protocol.protocol_sha256,
        "git_revision": _git_revision(),
        "mode": args.mode,
        "train_hours": args.train_hours,
        "lora_target": target.value,
        "training": asdict(training),
        "lora": lora_config,
        "whisper": whisper_config,
        "target_module_names": components.target_module_names,
        "trainable_parameters": components.trainable_parameters,
        "total_parameters": components.total_parameters,
        "baseline_summary_path": str(Path(baseline_path).resolve()),
        "baseline_summary_sha256": canonical_sha256(baseline_summary),
        "baseline_clean_cer": baseline_clean_cer,
        "test_split_accessed": False,
    }
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_revision": run_config["git_revision"],
        "cuda": torch.version.cuda,
        "device": device_properties.name,
        "packages": _versions(("torch", "transformers", "peft", "accelerate")),
    }
    data_audit = dataset.audit()
    data_audit.update(
        {
            "dev_manifest_path": str(dev_manifest.resolve()),
            "dev_manifest_sha256": canonical_sha256(dev_rows),
            "dev_utterances": len(dev_rows),
            "w0_baseline_clean_cer": baseline_clean_cer,
        }
    )
    reporter = TrainingReporter(
        console=ConsoleTrainingReporter(
            every_steps=int(
                lora_config["logging"]["console_interval_steps"]
                if args.console_every is None
                else args.console_every
            )
        ),
        structured=StructuredTrainingLogger(output_dir),
        structured_every_steps=int(
            lora_config["logging"]["structured_interval_steps"]
        ),
    )
    reporter.start(
        overview,
        run_config=run_config,
        environment=environment,
        data_audit=data_audit,
    )
    result = run_lora_training(
        model=components.model,
        dataset=dataset,
        collator=collator,
        config=training,
        evaluator=evaluator,
        selector=CheckpointSelector(
            baseline_clean_cer=baseline_clean_cer,
            maximum_clean_degradation_pp=float(
                lora_config["logging"]["maximum_clean_cer_degradation_pp"]
            ),
        ),
        reporter=reporter,
        output_dir=output_dir,
        device="cuda",
        resume_from=args.resume_from,
    )
    print(
        json.dumps(
            {
                "status": result.completion.status,
                "best_epoch": result.completion.best_epoch,
                "best_reverb_cer": result.completion.best_reverb_cer,
                "checkpoint_path": str(result.completion.checkpoint_path),
                "optimizer_steps": result.optimizer_steps,
                "latest_state_path": str(result.latest_state_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
