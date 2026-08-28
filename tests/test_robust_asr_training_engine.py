from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robust_asr.training.engine import (
    EpochEvaluation,
    TrainingConfig,
    run_lora_training,
)
from robust_asr.training.selection import (
    CheckpointSelector,
    DevCheckpointMetrics,
)


def test_training_config_rejects_invalid_parallel_settings() -> None:
    with pytest.raises(ValueError, match="prefetch_factor"):
        TrainingConfig(prefetch_factor=0)
    with pytest.raises(ValueError, match="num_workers"):
        TrainingConfig(num_workers=-1)


def test_cpu_training_loop_accumulates_selects_and_saves(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.tensor(0.5))
            self.config = SimpleNamespace(use_cache=True)

        def forward(self, *, input_features, labels):
            prediction = self.adapter * input_features.float().mean()
            target = labels.float().mean()
            return SimpleNamespace(loss=(prediction - target).square())

        def save_pretrained(self, path, *, safe_serialization):
            assert safe_serialization is True
            Path(path, "adapter.txt").write_text(
                str(float(self.adapter.detach())), encoding="utf-8"
            )

    rows = [
        {
            "input_features": torch.tensor([float(index + 1)]),
            "labels": torch.tensor([1]),
        }
        for index in range(5)
    ]

    def collate(batch):
        return {
            "input_features": torch.stack(
                [row["input_features"] for row in batch]
            ),
            "labels": torch.stack([row["labels"] for row in batch]),
        }

    class Reporter:
        def __init__(self) -> None:
            self.progress_rows = []
            self.evaluations = []
            self.completion = None

        def progress(self, value):
            self.progress_rows.append(value)

        def evaluation(self, value):
            self.evaluations.append(value)

        def predictions(self, *, epoch, rows):
            tuple(rows)

        def warning(self, *, code, message, context=None):
            raise AssertionError(f"unexpected warning: {code} {message} {context}")

        def complete(self, value):
            self.completion = value

    reporter = Reporter()

    def evaluate(model, epoch):
        return EpochEvaluation(
            metrics=DevCheckpointMetrics(
                epoch=epoch,
                clean_cer=0.10,
                reverb_cer=0.20 - 0.05 * (epoch - 1),
                heavy_cer=0.25 - 0.05 * (epoch - 1),
            ),
            per_rt60_cer={0.8: 0.25, 1.0: 0.30},
        )

    result = run_lora_training(
        model=ToyModel(),
        dataset=rows,
        collator=collate,
        config=TrainingConfig(
            epochs=2,
            per_device_batch_size=2,
            gradient_accumulation_steps=2,
            num_workers=0,
            precision="fp32",
        ),
        evaluator=evaluate,
        selector=CheckpointSelector(baseline_clean_cer=0.10),
        reporter=reporter,
        output_dir=tmp_path,
        device="cpu",
    )

    assert result.optimizer_steps == 4
    assert result.completed_epochs == 2
    assert result.latest_state_path.is_file()
    assert result.completion.best_epoch == 2
    assert (tmp_path / "checkpoints" / "epoch_002" / "adapter.txt").is_file()
    assert reporter.completion == result.completion
