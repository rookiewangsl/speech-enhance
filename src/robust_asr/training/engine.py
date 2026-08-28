"""Deterministic single-GPU LoRA optimization and epoch-level recovery."""

from __future__ import annotations

import math
import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from robust_asr.training.benchmark import optimizer_steps_per_epoch
from robust_asr.training.reporting import (
    EvaluationSummary,
    TrainingCompletion,
    TrainingProgress,
    TrainingReporter,
)
from robust_asr.training.selection import (
    CheckpointSelector,
    DevCheckpointMetrics,
)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 2026
    num_workers: int = 16
    prefetch_factor: int = 4
    ema_decay: float = 0.95
    precision: Literal["fp16", "fp32"] = "fp16"
    maximum_optimizer_steps: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "prefetch_factor",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if self.maximum_optimizer_steps is not None and (
            not isinstance(self.maximum_optimizer_steps, int)
            or isinstance(self.maximum_optimizer_steps, bool)
            or self.maximum_optimizer_steps <= 0
        ):
            raise ValueError("maximum_optimizer_steps must be a positive integer")
        for name in ("learning_rate", "max_grad_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.precision not in {"fp16", "fp32"}:
            raise ValueError("precision must be fp16 or fp32")


@dataclass(frozen=True)
class EpochEvaluation:
    metrics: DevCheckpointMetrics
    per_rt60_cer: Mapping[float, float]
    predictions: Sequence[Mapping[str, Any]] = ()
    substitutions: int | None = None
    deletions: int | None = None
    insertions: int | None = None

    def __post_init__(self) -> None:
        for rt60, cer in self.per_rt60_cer.items():
            if not math.isfinite(float(rt60)) or float(rt60) <= 0:
                raise ValueError("evaluation RT60 must be finite and positive")
            if not math.isfinite(float(cer)) or float(cer) < 0:
                raise ValueError("evaluation CER must be finite and non-negative")


@dataclass(frozen=True)
class TrainingResult:
    completion: TrainingCompletion
    optimizer_steps: int
    completed_epochs: int
    latest_state_path: Path


def _trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _load_trainable_state(model: Any, state: Mapping[str, Any]) -> None:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(parameters) != set(state):
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        raise ValueError(
            "resume LoRA parameter names disagree: "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    for name, parameter in parameters.items():
        source = state[name]
        if tuple(source.shape) != tuple(parameter.shape):
            raise ValueError(f"resume LoRA parameter shape mismatch: {name}")
        parameter.data.copy_(source.to(parameter.device, dtype=parameter.dtype))


def _atomic_torch_save(torch: Any, path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _save_adapter_atomic(model: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _loader_kwargs(
    *,
    dataset: Any,
    collator: Any,
    config: TrainingConfig,
    generator: Any,
    pin_memory: bool,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.per_device_batch_size,
        "shuffle": True,
        "collate_fn": collator,
        "num_workers": config.num_workers,
        "persistent_workers": False,
        "pin_memory": pin_memory,
        "generator": generator,
        "drop_last": False,
    }
    if config.num_workers > 0:
        values["prefetch_factor"] = config.prefetch_factor
    return values


def run_lora_training(
    *,
    model: Any,
    dataset: Any,
    collator: Any,
    config: TrainingConfig,
    evaluator: Callable[[Any, int], EpochEvaluation],
    selector: CheckpointSelector,
    reporter: TrainingReporter,
    output_dir: str | Path,
    device: str = "cuda",
    resume_from: str | Path | None = None,
) -> TrainingResult:
    """Train LoRA parameters and select dev checkpoints without touching test."""

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover - optional training stack
        raise RuntimeError("LoRA training requires torch and transformers") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if config.precision == "fp16" and device != "cuda":
        raise ValueError("the formal fp16 protocol requires CUDA")
    if len(dataset) <= 0:
        raise ValueError("training dataset is empty")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model.to(device)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    parameters = [value for value in model.parameters() if value.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    per_epoch = optimizer_steps_per_epoch(
        len(dataset),
        per_device_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    planned_steps = per_epoch * config.epochs
    if config.maximum_optimizer_steps is not None:
        planned_steps = min(planned_steps, config.maximum_optimizer_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(planned_steps * config.warmup_ratio),
        num_training_steps=planned_steps,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device == "cuda" and config.precision == "fp16")
    )

    start_epoch = 1
    global_step = 0
    best_checkpoint: Path | None = None
    if resume_from is not None:
        state = torch.load(Path(resume_from), map_location="cpu", weights_only=False)
        if int(state.get("schema_version", -1)) != 1:
            raise ValueError("unsupported training-state schema")
        _load_trainable_state(model, state["trainable_parameters"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        global_step = int(state["global_step"])
        start_epoch = int(state["completed_epoch"]) + 1
        selector_state = state.get("selector")
        if not isinstance(selector_state, Mapping):
            raise ValueError("resume state has no checkpoint-selector identity")
        if (
            float(selector_state.get("baseline_clean_cer", -1))
            != selector.baseline_clean_cer
            or float(selector_state.get("maximum_clean_degradation_pp", -1))
            != selector.maximum_clean_degradation_pp
        ):
            raise ValueError("resume checkpoint-selector protocol disagrees")
        raw_best = selector_state.get("best")
        if raw_best is not None:
            selector.best = DevCheckpointMetrics(**raw_best)
        raw_checkpoint = state.get("best_checkpoint")
        if raw_checkpoint is not None:
            best_checkpoint = Path(str(raw_checkpoint))
            if not best_checkpoint.is_dir():
                raise FileNotFoundError(best_checkpoint)
        if start_epoch > config.epochs or global_step >= planned_steps:
            raise ValueError("resume state has already exhausted this training budget")

    optimizer.zero_grad(set_to_none=True)
    ema_loss: float | None = None
    started = time.perf_counter()
    completed_epochs = start_epoch - 1
    latest_state_path = destination / "latest_state.pt"

    for epoch in range(start_epoch, config.epochs + 1):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch - 1)
        generator = torch.Generator()
        generator.manual_seed(config.seed + epoch - 1)
        loader = DataLoader(
            **_loader_kwargs(
                dataset=dataset,
                collator=collator,
                config=config,
                generator=generator,
                pin_memory=device == "cuda",
            )
        )
        micro_batches = len(loader)
        epoch_step = 0
        group_progress = 0
        group_size = min(config.gradient_accumulation_steps, micro_batches)
        for micro_index, batch in enumerate(loader, start=1):
            features = batch["input_features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if config.precision == "fp16"
                else nullcontext()
            )
            with autocast:
                output = model(input_features=features, labels=labels)
                loss = output.loss
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite Whisper training loss")
            loss_value = float(loss.detach().item())
            ema_loss = (
                loss_value
                if ema_loss is None
                else config.ema_decay * ema_loss
                + (1.0 - config.ema_decay) * loss_value
            )
            scaler.scale(loss / group_size).backward()
            group_progress += 1
            if group_progress < group_size:
                continue

            scaler.unscale_(optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                parameters, config.max_grad_norm
            )
            grad_norm = float(grad_norm_tensor.detach().item())
            if not math.isfinite(grad_norm):
                raise FloatingPointError("non-finite LoRA gradient norm")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            epoch_step += 1
            group_progress = 0
            remaining_micro_batches = micro_batches - micro_index
            group_size = min(
                config.gradient_accumulation_steps,
                remaining_micro_batches,
            )

            elapsed = time.perf_counter() - started
            speed = global_step / max(elapsed, 1e-9)
            reporter.progress(
                TrainingProgress(
                    epoch=epoch,
                    total_epochs=config.epochs,
                    step=epoch_step,
                    steps_per_epoch=min(
                        per_epoch,
                        planned_steps - (epoch - 1) * per_epoch,
                    ),
                    loss=loss_value,
                    ema_loss=float(ema_loss),
                    learning_rate=float(scheduler.get_last_lr()[0]),
                    grad_norm=grad_norm,
                    steps_per_second=speed,
                    gpu_memory_gib=(
                        float(torch.cuda.max_memory_allocated()) / 1024**3
                        if device == "cuda"
                        else 0.0
                    ),
                    eta_seconds=(planned_steps - global_step) / max(speed, 1e-9),
                )
            )
            if global_step >= planned_steps:
                break

        evaluation = evaluator(model, epoch)
        if evaluation.metrics.epoch != epoch:
            raise ValueError("evaluator returned metrics for another epoch")
        decision = selector.consider(evaluation.metrics)
        checkpoint_path: Path | None = None
        if decision.improved:
            checkpoint_path = destination / "checkpoints" / f"epoch_{epoch:03d}"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            _save_adapter_atomic(model, checkpoint_path)
            best_checkpoint = checkpoint_path
        elif not decision.eligible:
            reporter.warning(
                code="CLEAN_CER_SAFETY_GATE",
                message=(
                    f"epoch {epoch} clean CER degraded by "
                    f"{decision.clean_degradation_pp:.3f} percentage points"
                ),
                context={"epoch": epoch, "decision": decision.__dict__},
            )
        best_reverb = (
            evaluation.metrics.reverb_cer
            if decision.best_reverb_cer is None
            else decision.best_reverb_cer
        )
        reporter.evaluation(
            EvaluationSummary(
                epoch=epoch,
                total_epochs=config.epochs,
                clean_cer=evaluation.metrics.clean_cer,
                reverb_cer=evaluation.metrics.reverb_cer,
                heavy_cer=evaluation.metrics.heavy_cer,
                best_reverb_cer=best_reverb,
                improved=decision.improved,
                checkpoint_path=checkpoint_path,
                per_rt60_cer=evaluation.per_rt60_cer,
                substitutions=evaluation.substitutions,
                deletions=evaluation.deletions,
                insertions=evaluation.insertions,
            )
        )
        reporter.predictions(epoch=epoch, rows=evaluation.predictions)
        completed_epochs = epoch
        state = {
            "schema_version": 1,
            "completed_epoch": epoch,
            "global_step": global_step,
            "trainable_parameters": _trainable_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "selector": {
                "baseline_clean_cer": selector.baseline_clean_cer,
                "maximum_clean_degradation_pp": (
                    selector.maximum_clean_degradation_pp
                ),
                "best": None if selector.best is None else selector.best.__dict__,
            },
            "best_checkpoint": (
                None if best_checkpoint is None else str(best_checkpoint)
            ),
        }
        _atomic_torch_save(torch, latest_state_path, state)
        if global_step >= planned_steps:
            break
        model.train()

    if selector.best is None or best_checkpoint is None:
        raise RuntimeError("no checkpoint passed the clean CER safety gate")
    completion = TrainingCompletion(
        best_epoch=selector.best.epoch,
        best_reverb_cer=selector.best.reverb_cer,
        elapsed_seconds=time.perf_counter() - started,
        peak_gpu_memory_gib=(
            float(torch.cuda.max_memory_allocated()) / 1024**3
            if device == "cuda"
            else 0.0
        ),
        checkpoint_path=best_checkpoint,
    )
    reporter.complete(completion)
    return TrainingResult(
        completion=completion,
        optimizer_steps=global_step,
        completed_epochs=completed_epochs,
        latest_state_path=latest_state_path,
    )
