"""Deterministic dev checkpoint selection with a clean-CER safety gate."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DevCheckpointMetrics:
    epoch: int
    clean_cer: float
    reverb_cer: float
    heavy_cer: float

    def __post_init__(self) -> None:
        if not isinstance(self.epoch, int) or isinstance(self.epoch, bool) or self.epoch <= 0:
            raise ValueError("epoch must be a positive integer")
        for name in ("clean_cer", "reverb_cer", "heavy_cer"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class CheckpointDecision:
    eligible: bool
    improved: bool
    clean_degradation_pp: float
    best_epoch: int | None
    best_reverb_cer: float | None
    reason: str


class CheckpointSelector:
    """Select lowest reverb CER subject to a frozen clean degradation limit."""

    def __init__(
        self,
        *,
        baseline_clean_cer: float,
        maximum_clean_degradation_pp: float = 0.5,
    ) -> None:
        if not math.isfinite(baseline_clean_cer) or baseline_clean_cer < 0:
            raise ValueError("baseline_clean_cer must be finite and non-negative")
        if (
            not math.isfinite(maximum_clean_degradation_pp)
            or maximum_clean_degradation_pp < 0
        ):
            raise ValueError(
                "maximum_clean_degradation_pp must be finite and non-negative"
            )
        self.baseline_clean_cer = baseline_clean_cer
        self.maximum_clean_degradation_pp = maximum_clean_degradation_pp
        self.best: DevCheckpointMetrics | None = None

    def consider(self, metrics: DevCheckpointMetrics) -> CheckpointDecision:
        degradation_pp = 100.0 * (
            metrics.clean_cer - self.baseline_clean_cer
        )
        eligible = degradation_pp <= self.maximum_clean_degradation_pp + 1e-12
        improved = eligible and (
            self.best is None or metrics.reverb_cer < self.best.reverb_cer
        )
        if improved:
            self.best = metrics
        if not eligible:
            reason = "clean_cer_safety_gate_failed"
        elif improved:
            reason = "lowest_eligible_dev_reverb_cer"
        else:
            reason = "eligible_but_not_better"
        return CheckpointDecision(
            eligible=eligible,
            improved=improved,
            clean_degradation_pp=degradation_pp,
            best_epoch=None if self.best is None else self.best.epoch,
            best_reverb_cer=(
                None if self.best is None else self.best.reverb_cer
            ),
            reason=reason,
        )
