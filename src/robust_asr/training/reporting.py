"""Concise terminal progress and detailed structured training logs.

The terminal is deliberately limited to information needed to judge convergence,
runtime, and resource pressure. Reproducibility metadata and condition-level
metrics are written to files under the run directory instead.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO


def _finite_nonnegative(value: float, *, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _positive_int(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("structured logs cannot contain NaN or infinity")
    return value


def _serialized_json(value: Any, *, pretty: bool) -> str:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_serialized_json(value, pretty=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_identity_json(path: Path, value: Mapping[str, Any]) -> None:
    normalized = _json_ready(value)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError(f"refusing to overwrite incompatible run metadata: {path}")
        return
    _write_json_atomic(path, normalized)


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_serialized_json(row, pretty=False))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"structured log row {line_number} is not an object")
        rows.append(value)
    return rows


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(_serialized_json(row, pretty=False))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, final_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{final_seconds:02d}"
    return f"{minutes:02d}:{final_seconds:02d}"


@dataclass(frozen=True)
class RunOverview:
    experiment_id: str
    model_name: str
    lora_target: str
    lora_rank: int
    trainable_ratio: float
    train_hours: float
    train_utterances: int
    dev_utterances: int
    clean_probability: float
    reverb_probability: float
    precision: str
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    epochs: int
    device_name: str
    device_memory_gib: float
    output_dir: Path

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.model_name or not self.lora_target:
            raise ValueError("run identifiers must be non-empty")
        for name in (
            "lora_rank",
            "train_utterances",
            "dev_utterances",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "epochs",
        ):
            _positive_int(getattr(self, name), name=name)
        for name in (
            "trainable_ratio",
            "train_hours",
            "clean_probability",
            "reverb_probability",
            "learning_rate",
            "device_memory_gib",
        ):
            _finite_nonnegative(float(getattr(self, name)), name=name)
        if abs(self.clean_probability + self.reverb_probability - 1.0) > 1e-12:
            raise ValueError("clean and reverb probabilities must sum to one")
        if self.trainable_ratio > 1.0:
            raise ValueError("trainable_ratio cannot exceed one")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class TrainingProgress:
    epoch: int
    total_epochs: int
    step: int
    steps_per_epoch: int
    loss: float
    ema_loss: float
    learning_rate: float
    grad_norm: float
    steps_per_second: float
    gpu_memory_gib: float
    eta_seconds: float

    def __post_init__(self) -> None:
        for name in ("epoch", "total_epochs", "step", "steps_per_epoch"):
            _positive_int(getattr(self, name), name=name)
        if self.epoch > self.total_epochs:
            raise ValueError("epoch cannot exceed total_epochs")
        if self.step > self.steps_per_epoch:
            raise ValueError("step cannot exceed steps_per_epoch")
        for name in (
            "loss",
            "ema_loss",
            "learning_rate",
            "grad_norm",
            "steps_per_second",
            "gpu_memory_gib",
            "eta_seconds",
        ):
            _finite_nonnegative(float(getattr(self, name)), name=name)


@dataclass(frozen=True)
class EvaluationSummary:
    epoch: int
    total_epochs: int
    clean_cer: float
    reverb_cer: float
    heavy_cer: float
    best_reverb_cer: float
    improved: bool
    checkpoint_path: Path | None = None
    per_rt60_cer: Mapping[float, float] | None = None
    substitutions: int | None = None
    deletions: int | None = None
    insertions: int | None = None

    def __post_init__(self) -> None:
        _positive_int(self.epoch, name="epoch")
        _positive_int(self.total_epochs, name="total_epochs")
        if self.epoch > self.total_epochs:
            raise ValueError("epoch cannot exceed total_epochs")
        for name in ("clean_cer", "reverb_cer", "heavy_cer", "best_reverb_cer"):
            _finite_nonnegative(float(getattr(self, name)), name=name)
        for rt60, cer in (self.per_rt60_cer or {}).items():
            _finite_nonnegative(float(rt60), name="rt60")
            _finite_nonnegative(float(cer), name="per_rt60_cer")
        for name in ("substitutions", "deletions", "insertions"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class TrainingCompletion:
    best_epoch: int
    best_reverb_cer: float
    elapsed_seconds: float
    peak_gpu_memory_gib: float
    checkpoint_path: Path
    status: str = "SUCCESS"

    def __post_init__(self) -> None:
        _positive_int(self.best_epoch, name="best_epoch")
        for name in ("best_reverb_cer", "elapsed_seconds", "peak_gpu_memory_gib"):
            _finite_nonnegative(float(getattr(self, name)), name=name)
        if not self.status:
            raise ValueError("status must be non-empty")


class ConsoleTrainingReporter:
    """Render only live progress and hyperparameter-tuning signals."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        every_steps: int = 20,
        live: bool | None = None,
    ) -> None:
        _positive_int(every_steps, name="every_steps")
        self.stream = stream or sys.stderr
        self.every_steps = every_steps
        self.live = bool(getattr(self.stream, "isatty", lambda: False)()) if live is None else live
        self._progress_active = False

    def _finish_progress(self) -> None:
        if self._progress_active:
            self.stream.write("\n")
            self.stream.flush()
            self._progress_active = False

    def _line(self, value: str) -> None:
        self._finish_progress()
        self.stream.write(value + "\n")
        self.stream.flush()

    def start_run(self, overview: RunOverview) -> None:
        condition = (
            f"MCT clean:reverb={overview.clean_probability:g}:"
            f"{overview.reverb_probability:g}"
            if overview.reverb_probability > 0
            else "clean-only"
        )
        self._line(f"Run: {overview.experiment_id}")
        self._line(
            f"Model: {overview.model_name} | LoRA: {overview.lora_target}, "
            f"r={overview.lora_rank}, trainable={overview.trainable_ratio:.3%}"
        )
        self._line(
            f"Data: train={overview.train_hours:.1f}h/{overview.train_utterances} utt "
            f"| dev={overview.dev_utterances} utt | {condition}"
        )
        self._line(
            f"Train: {overview.precision} | batch={overview.per_device_batch_size}×"
            f"{overview.gradient_accumulation_steps}={overview.effective_batch_size} "
            f"| lr={overview.learning_rate:.1e} | epochs={overview.epochs}"
        )
        self._line(
            f"GPU: {overview.device_name} {overview.device_memory_gib:.1f}GB "
            f"| Output: {overview.output_dir}"
        )

    def progress(self, value: TrainingProgress) -> None:
        if (
            value.step != 1
            and value.step != value.steps_per_epoch
            and value.step % self.every_steps != 0
        ):
            return
        percent = 100.0 * value.step / value.steps_per_epoch
        line = (
            f"Epoch {value.epoch}/{value.total_epochs} "
            f"{value.step}/{value.steps_per_epoch} [{percent:3.0f}%] "
            f"| loss {value.loss:.4f} (ema {value.ema_loss:.4f}) "
            f"| lr {value.learning_rate:.2e} | grad {value.grad_norm:.2f} "
            f"| {value.steps_per_second:.2f} step/s "
            f"| GPU {value.gpu_memory_gib:.1f}GB | ETA {_duration(value.eta_seconds)}"
        )
        if self.live:
            self.stream.write("\r\033[2K" + line)
            self.stream.flush()
            self._progress_active = True
        else:
            self._line(line)

    def evaluation(self, value: EvaluationSummary) -> None:
        direction = "↓" if value.improved else "—"
        checkpoint = "checkpoint saved" if value.checkpoint_path else "checkpoint unchanged"
        self._line(
            f"Eval {value.epoch}/{value.total_epochs} | clean CER {value.clean_cer:.2%} "
            f"| reverb CER {value.reverb_cer:.2%} | heavy CER {value.heavy_cer:.2%} "
            f"| best {value.best_reverb_cer:.2%} {direction} | {checkpoint}"
        )

    def warning(self, code: str, message: str) -> None:
        if not code or not message:
            raise ValueError("warning code and message must be non-empty")
        self._line(f"WARNING [{code}] {message}")

    def complete(self, value: TrainingCompletion) -> None:
        self._line(
            f"Training complete | best epoch={value.best_epoch} "
            f"| best reverb CER={value.best_reverb_cer:.2%} "
            f"| time={_duration(value.elapsed_seconds)} "
            f"| peak GPU={value.peak_gpu_memory_gib:.1f}GB "
            f"| checkpoint={value.checkpoint_path} | status={value.status}"
        )


class StructuredTrainingLogger:
    """Persist reproducibility metadata and complete condition-level metrics."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._progress_keys = {
            (int(row["epoch"]), int(row["step"]))
            for row in _read_jsonl(self.output_dir / "train_metrics.jsonl")
        }
        self._evaluation_epochs = {
            int(row["epoch"])
            for row in _read_jsonl(self.output_dir / "eval_metrics.jsonl")
        }
        self._prediction_epochs = {
            int(row["epoch"])
            for row in _read_jsonl(self.output_dir / "predictions.jsonl")
        }

    def start(
        self,
        *,
        run_config: Mapping[str, Any],
        environment: Mapping[str, Any],
        data_audit: Mapping[str, Any],
    ) -> None:
        _write_identity_json(self.output_dir / "run_config.json", run_config)
        _write_identity_json(self.output_dir / "environment.json", environment)
        _write_identity_json(self.output_dir / "data_audit.json", data_audit)

    def progress(self, value: TrainingProgress) -> None:
        row = {"recorded_at_utc": _utc_now(), **asdict(value)}
        path = self.output_dir / "train_metrics.jsonl"
        key = (value.epoch, value.step)
        if key in self._progress_keys:
            existing = [
                previous
                for previous in _read_jsonl(path)
                if (int(previous["epoch"]), int(previous["step"])) != key
            ]
            _write_jsonl_atomic(path, (*existing, row))
        else:
            _append_jsonl(path, (row,))
            self._progress_keys.add(key)

    def evaluation(self, value: EvaluationSummary) -> None:
        row = {"recorded_at_utc": _utc_now(), **asdict(value)}
        metrics_path = self.output_dir / "eval_metrics.jsonl"
        if value.epoch in self._evaluation_epochs:
            existing = [
                previous
                for previous in _read_jsonl(metrics_path)
                if int(previous["epoch"]) != value.epoch
            ]
            _write_jsonl_atomic(metrics_path, (*existing, row))
        else:
            _append_jsonl(metrics_path, (row,))
            self._evaluation_epochs.add(value.epoch)

        path = self.output_dir / "eval_by_rt60.json"
        if path.exists():
            history = json.loads(path.read_text(encoding="utf-8"))
        else:
            history = {"schema_version": 1, "epochs": {}}
        history["epochs"][str(value.epoch)] = {
            str(rt60): cer for rt60, cer in sorted((value.per_rt60_cer or {}).items())
        }
        _write_json_atomic(path, history)

    def predictions(
        self, *, epoch: int, rows: Iterable[Mapping[str, Any]]
    ) -> None:
        _positive_int(epoch, name="epoch")
        recorded_at = _utc_now()
        enriched = tuple(
            {**dict(row), "recorded_at_utc": recorded_at, "epoch": epoch}
            for row in rows
        )
        path = self.output_dir / "predictions.jsonl"
        if epoch in self._prediction_epochs:
            existing = [
                previous
                for previous in _read_jsonl(path)
                if int(previous["epoch"]) != epoch
            ]
            _write_jsonl_atomic(path, (*existing, *enriched))
        else:
            _append_jsonl(path, enriched)
            self._prediction_epochs.add(epoch)

    def warning(
        self,
        *,
        code: str,
        message: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if not code or not message:
            raise ValueError("warning code and message must be non-empty")
        row = {
            "recorded_at_utc": _utc_now(),
            "code": code,
            "message": message,
            "context": dict(context or {}),
        }
        _append_jsonl(self.output_dir / "warnings.jsonl", (row,))

    def complete(self, value: TrainingCompletion) -> None:
        summary = {"finished_at_utc": _utc_now(), **asdict(value)}
        _write_json_atomic(self.output_dir / "training_summary.json", summary)


class TrainingReporter:
    """Keep terminal and structured logging policies synchronized."""

    def __init__(
        self,
        *,
        console: ConsoleTrainingReporter,
        structured: StructuredTrainingLogger,
        structured_every_steps: int = 10,
    ) -> None:
        _positive_int(structured_every_steps, name="structured_every_steps")
        self.console = console
        self.structured = structured
        self.structured_every_steps = structured_every_steps

    def start(
        self,
        overview: RunOverview,
        *,
        run_config: Mapping[str, Any],
        environment: Mapping[str, Any],
        data_audit: Mapping[str, Any],
    ) -> None:
        self.structured.start(
            run_config=run_config,
            environment=environment,
            data_audit=data_audit,
        )
        self.console.start_run(overview)

    def progress(self, value: TrainingProgress) -> None:
        if (
            value.step == 1
            or value.step == value.steps_per_epoch
            or value.step % self.structured_every_steps == 0
        ):
            self.structured.progress(value)
        self.console.progress(value)

    def evaluation(self, value: EvaluationSummary) -> None:
        self.structured.evaluation(value)
        self.console.evaluation(value)

    def predictions(
        self, *, epoch: int, rows: Iterable[Mapping[str, Any]]
    ) -> None:
        self.structured.predictions(epoch=epoch, rows=rows)

    def warning(
        self,
        *,
        code: str,
        message: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.structured.warning(code=code, message=message, context=context)
        self.console.warning(code, message)

    def complete(self, value: TrainingCompletion) -> None:
        self.structured.complete(value)
        self.console.complete(value)
