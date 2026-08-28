#!/usr/bin/env python3
"""Run the transient 100-step Whisper LoRA hardware gate on CUDA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

from robust_asr.config import canonical_sha256, load_json_object
from robust_asr.lora import LoRAProtocol, LoRATarget, choose_training_budget
from robust_asr.manifest import read_jsonl
from robust_asr.models.whisper_lora import load_whisper_lora_components
from robust_asr.paths import require_data_root
from robust_asr.training.benchmark import (
    BenchmarkConfig,
    optimizer_steps_per_epoch,
    run_lora_optimizer_benchmark,
)
from robust_asr.training.data import WhisperAdaptationDataset, WhisperBatchCollator
from robust_asr.training.reporting import (
    ConsoleTrainingReporter,
    RunOverview,
    StructuredTrainingLogger,
    TrainingReporter,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--mode", choices=("clean", "mct"), default="clean")
    parser.add_argument(
        "--lora-target",
        choices=tuple(value.value for value in LoRATarget),
        default=LoRATarget.ENCODER_QV.value,
    )
    parser.add_argument("--optimizer-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--console-every", type=int, default=20)
    parser.add_argument("--output-name")
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


def _write_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    lora_config = load_json_object(Path("configs/robust_asr/lora.json"))
    whisper_config = load_json_object(Path("configs/robust_asr/whisper.json"))
    target = LoRATarget(args.lora_target)
    benchmark = BenchmarkConfig(
        optimizer_steps=args.optimizer_steps,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=float(lora_config["learning_rate"]),
        weight_decay=float(lora_config["weight_decay"]),
        warmup_ratio=float(lora_config["warmup_ratio"]),
        max_grad_norm=float(lora_config["max_grad_norm"]),
        seed=int(lora_config["seed"]),
        num_workers=args.num_workers,
    )
    default_name = (
        f"benchmark_{args.mode}_{target.value}_"
        f"b{benchmark.per_device_batch_size}_a"
        f"{benchmark.gradient_accumulation_steps}"
    )
    output_name = args.output_name or default_name
    if Path(output_name).name != output_name:
        raise ValueError("--output-name must be a basename")
    output_dir = root / "runs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        root / "manifests" / "aishell1" / "aishell1_train_20h.jsonl"
    )
    corpus_root = root / "corpora" / "aishell1"
    dataset = WhisperAdaptationDataset(
        manifest_path=manifest_path,
        corpus_root=corpus_root,
        mode=args.mode,
        rir_manifest_path=(
            None
            if args.mode == "clean"
            else root / "rir" / "pyroom_v1" / "train.jsonl"
        ),
        rir_root=(
            None if args.mode == "clean" else root / "rir" / "pyroom_v1"
        ),
        reverb_probability=0.5,
        seed=benchmark.seed,
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

    import torch

    train_rows = read_jsonl(manifest_path)
    train_hours = sum(float(row["duration_seconds"]) for row in train_rows) / 3600
    dev_rows = read_jsonl(
        root / "manifests" / "aishell1" / "aishell1_dev_model.jsonl"
    )
    device_properties = torch.cuda.get_device_properties(0)
    ratio = components.trainable_parameters / components.total_parameters
    overview = RunOverview(
        experiment_id=output_name,
        model_name=str(whisper_config["model_id"]),
        lora_target=target.value,
        lora_rank=int(lora_config["rank"]),
        trainable_ratio=ratio,
        train_hours=train_hours,
        train_utterances=len(dataset),
        dev_utterances=len(dev_rows),
        clean_probability=1.0 if args.mode == "clean" else 0.5,
        reverb_probability=0.0 if args.mode == "clean" else 0.5,
        precision="fp16",
        per_device_batch_size=benchmark.per_device_batch_size,
        gradient_accumulation_steps=benchmark.gradient_accumulation_steps,
        learning_rate=benchmark.learning_rate,
        epochs=1,
        device_name=device_properties.name,
        device_memory_gib=device_properties.total_memory / 1024**3,
        output_dir=output_dir,
    )
    run_config = {
        "schema_version": 1,
        "purpose": "transient_hardware_benchmark_no_checkpoint",
        "mode": args.mode,
        "lora_target": target.value,
        "benchmark": benchmark.__dict__,
        "lora": lora_config,
        "whisper": whisper_config,
        "target_module_names": components.target_module_names,
        "trainable_parameters": components.trainable_parameters,
        "total_parameters": components.total_parameters,
    }
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "device": device_properties.name,
        "packages": _versions(("torch", "transformers", "peft", "accelerate")),
    }
    data_audit = dataset.audit()
    data_audit["dev_model_utterances"] = len(dev_rows)
    data_audit["run_identity_sha256"] = canonical_sha256(
        {"run_config": run_config, "data_audit": data_audit}
    )
    reporter = TrainingReporter(
        console=ConsoleTrainingReporter(every_steps=args.console_every),
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
    result = run_lora_optimizer_benchmark(
        model=components.model,
        dataset=dataset,
        collator=collator,
        config=benchmark,
        progress_callback=reporter.progress,
    )
    steps_per_epoch = optimizer_steps_per_epoch(
        len(dataset),
        per_device_batch_size=benchmark.per_device_batch_size,
        gradient_accumulation_steps=benchmark.gradient_accumulation_steps,
    )
    budget = choose_training_budget(
        seconds_per_optimizer_step=result.seconds_per_optimizer_step,
        optimizer_steps_per_epoch_20h=steps_per_epoch,
        maximum_run_hours=float(lora_config["maximum_run_hours"]),
    )
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "transient_hardware_benchmark_no_checkpoint",
        "result": result.as_dict(),
        "optimizer_steps_per_epoch_20h": steps_per_epoch,
        "formal_budget": budget.__dict__,
        "adapter_checkpoint_saved": False,
    }
    _write_atomic(output_dir / "benchmark_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
