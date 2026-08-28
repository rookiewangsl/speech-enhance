import pytest

from robust_asr.training.benchmark import (
    BenchmarkConfig,
    optimizer_steps_per_epoch,
)


def test_benchmark_config_freezes_effective_batch_size() -> None:
    config = BenchmarkConfig()

    assert config.optimizer_steps == 100
    assert config.effective_batch_size == 16


def test_optimizer_steps_per_epoch_uses_ceil_for_both_levels() -> None:
    assert optimizer_steps_per_epoch(
        33,
        per_device_batch_size=2,
        gradient_accumulation_steps=8,
    ) == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"optimizer_steps": 0}, "optimizer_steps"),
        ({"warmup_ratio": 1.1}, "warmup_ratio"),
        ({"num_workers": -1}, "num_workers"),
    ],
)
def test_benchmark_config_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        BenchmarkConfig(**kwargs)
