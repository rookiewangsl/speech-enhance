"""Single-GPU optimizer-step benchmark for the frozen Whisper LoRA protocol."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from robust_asr.training.reporting import TrainingProgress


@dataclass(frozen=True)
class BenchmarkConfig:
    optimizer_steps: int = 100
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 2026
    num_workers: int = 4
    ema_decay: float = 0.95

    def __post_init__(self) -> None:
        for name in (
            "optimizer_steps",
            "per_device_batch_size",
            "gradient_accumulation_steps",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
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

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class BenchmarkResult:
    optimizer_steps: int
    micro_batches: int
    elapsed_seconds: float
    seconds_per_optimizer_step: float
    audio_examples: int
    examples_per_second: float
    peak_gpu_memory_gib: float
    final_loss: float
    final_ema_loss: float
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    precision: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def optimizer_steps_per_epoch(
    dataset_size: int,
    *,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return ceil micro-batches/accumulation for one complete epoch."""

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if per_device_batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("batch sizes must be positive")
    micro_batches = math.ceil(dataset_size / per_device_batch_size)
    return math.ceil(micro_batches / gradient_accumulation_steps)


def run_lora_optimizer_benchmark(
    *,
    model: Any,
    dataset: Any,
    collator: Any,
    config: BenchmarkConfig,
    device: str = "cuda",
    progress_callback: Callable[[TrainingProgress], None] | None = None,
) -> BenchmarkResult:
    """Run transient FP16 updates and return timing/memory measurements.

    No checkpoint is saved. The caller must discard the model after this
    function so benchmark updates cannot be mistaken for a trained adapter.
    """

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover - optional training stack
        raise RuntimeError("LoRA benchmark requires torch and transformers") from exc
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen LoRA benchmark requires CUDA")

    torch.manual_seed(config.seed)
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
    warmup_steps = round(config.optimizer_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=config.optimizer_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        persistent_workers=False,
        pin_memory=True,
        generator=generator,
    )

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_batches = 0
    examples = 0
    final_loss = 0.0
    ema_loss: float | None = None
    started = time.perf_counter()
    data_iterator = iter(loader)
    while optimizer_step < config.optimizer_steps:
        try:
            batch = next(data_iterator)
        except StopIteration:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(1)
            generator.manual_seed(config.seed + 1)
            data_iterator = iter(loader)
            batch = next(data_iterator)
        features = batch["input_features"].to(
            device, non_blocking=True
        )
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(input_features=features, labels=labels)
            loss = output.loss
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite Whisper training loss")
        final_loss = float(loss.detach().item())
        ema_loss = (
            final_loss
            if ema_loss is None
            else config.ema_decay * ema_loss
            + (1.0 - config.ema_decay) * final_loss
        )
        scaler.scale(loss / config.gradient_accumulation_steps).backward()
        micro_batches += 1
        examples += int(features.shape[0])
        if micro_batches % config.gradient_accumulation_steps != 0:
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
        optimizer_step += 1

        elapsed = time.perf_counter() - started
        if progress_callback is not None:
            speed = optimizer_step / max(elapsed, 1e-9)
            progress_callback(
                TrainingProgress(
                    epoch=1,
                    total_epochs=1,
                    step=optimizer_step,
                    steps_per_epoch=config.optimizer_steps,
                    loss=final_loss,
                    ema_loss=float(ema_loss),
                    learning_rate=float(scheduler.get_last_lr()[0]),
                    grad_norm=grad_norm,
                    steps_per_second=speed,
                    gpu_memory_gib=float(torch.cuda.max_memory_allocated())
                    / 1024**3,
                    eta_seconds=(config.optimizer_steps - optimizer_step)
                    / max(speed, 1e-9),
                )
            )

    elapsed = time.perf_counter() - started
    return BenchmarkResult(
        optimizer_steps=optimizer_step,
        micro_batches=micro_batches,
        elapsed_seconds=elapsed,
        seconds_per_optimizer_step=elapsed / optimizer_step,
        audio_examples=examples,
        examples_per_second=examples / elapsed,
        peak_gpu_memory_gib=float(torch.cuda.max_memory_allocated()) / 1024**3,
        final_loss=final_loss,
        final_ema_loss=float(ema_loss),
        batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        effective_batch_size=config.effective_batch_size,
        precision="fp16",
    )
