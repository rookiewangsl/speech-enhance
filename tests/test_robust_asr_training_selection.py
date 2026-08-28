import pytest

from robust_asr.training.selection import (
    CheckpointSelector,
    DevCheckpointMetrics,
)


def _metrics(epoch: int, clean: float, reverb: float) -> DevCheckpointMetrics:
    return DevCheckpointMetrics(
        epoch=epoch,
        clean_cer=clean,
        reverb_cer=reverb,
        heavy_cer=reverb + 0.02,
    )


def test_checkpoint_selector_applies_clean_gate_before_reverb_metric() -> None:
    selector = CheckpointSelector(
        baseline_clean_cer=0.10,
        maximum_clean_degradation_pp=0.5,
    )

    first = selector.consider(_metrics(1, 0.104, 0.15))
    rejected = selector.consider(_metrics(2, 0.106, 0.10))
    improved = selector.consider(_metrics(3, 0.103, 0.14))

    assert first.improved is True
    assert rejected.eligible is False
    assert rejected.reason == "clean_cer_safety_gate_failed"
    assert rejected.best_epoch == 1
    assert improved.improved is True
    assert improved.best_epoch == 3


def test_checkpoint_selector_keeps_earlier_epoch_on_exact_tie() -> None:
    selector = CheckpointSelector(baseline_clean_cer=0.10)
    selector.consider(_metrics(1, 0.10, 0.14))

    decision = selector.consider(_metrics(2, 0.10, 0.14))

    assert decision.improved is False
    assert decision.best_epoch == 1


def test_checkpoint_selector_reports_degradation_in_percentage_points() -> None:
    selector = CheckpointSelector(baseline_clean_cer=0.10)

    decision = selector.consider(_metrics(1, 0.104, 0.14))

    assert decision.clean_degradation_pp == pytest.approx(0.4)
