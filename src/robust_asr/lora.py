"""LoRA target selection and single-GPU training budget decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class LoRATarget(str, Enum):
    ENCODER_QV = "encoder_qv"
    ENCODER_DECODER_QV = "encoder_decoder_qv"


@dataclass(frozen=True)
class LoRAProtocol:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.bias != "none":
            raise ValueError("v0.1 freezes all base-model biases")

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


def _is_qv_projection(name: str) -> bool:
    return name.endswith(".q_proj") or name.endswith(".v_proj")


def select_lora_target_names(
    module_names: Iterable[str],
    target: LoRATarget,
) -> tuple[str, ...]:
    """Select exact Whisper attention projections from model module names."""

    selected: list[str] = []
    for name in sorted(set(module_names)):
        if not _is_qv_projection(name):
            continue
        encoder = ".encoder.layers." in name and ".self_attn." in name
        decoder_self = ".decoder.layers." in name and ".self_attn." in name
        decoder_cross = ".decoder.layers." in name and ".encoder_attn." in name
        if target == LoRATarget.ENCODER_QV and encoder:
            selected.append(name)
        elif target == LoRATarget.ENCODER_DECODER_QV and (
            encoder or decoder_self or decoder_cross
        ):
            selected.append(name)
    if not selected:
        raise ValueError(f"no Whisper modules matched target {target.value}")
    return tuple(selected)


def expected_whisper_target_count(
    target: LoRATarget,
    *,
    encoder_layers: int = 12,
    decoder_layers: int = 12,
) -> int:
    """Expected q/v projection count for Whisper-small-style architecture."""

    if encoder_layers <= 0 or decoder_layers <= 0:
        raise ValueError("layer counts must be positive")
    encoder_count = encoder_layers * 2
    if target == LoRATarget.ENCODER_QV:
        return encoder_count
    # Decoder has q/v in self-attention and encoder-attention.
    return encoder_count + decoder_layers * 4


def validate_whisper_target_count(
    selected: Iterable[str],
    target: LoRATarget,
    *,
    encoder_layers: int = 12,
    decoder_layers: int = 12,
) -> None:
    actual = len(tuple(selected))
    expected = expected_whisper_target_count(
        target,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    )
    if actual != expected:
        raise ValueError(
            f"target {target.value} matched {actual} modules; expected {expected}"
        )


@dataclass(frozen=True)
class TrainingBudget:
    subset_hours: int
    epochs: int
    projected_hours: float
    reason: str


def choose_training_budget(
    *,
    seconds_per_optimizer_step: float,
    optimizer_steps_per_epoch_20h: int,
    maximum_run_hours: float = 8.0,
) -> TrainingBudget:
    """Apply the frozen 20h/3→20h/2→10h/2 runtime fallback."""

    if seconds_per_optimizer_step <= 0:
        raise ValueError("seconds_per_optimizer_step must be positive")
    if optimizer_steps_per_epoch_20h <= 0:
        raise ValueError("optimizer_steps_per_epoch_20h must be positive")
    if maximum_run_hours <= 0:
        raise ValueError("maximum_run_hours must be positive")

    def project(steps_per_epoch: float, epochs: int) -> float:
        return seconds_per_optimizer_step * steps_per_epoch * epochs / 3600.0

    three_epochs = project(optimizer_steps_per_epoch_20h, 3)
    if three_epochs <= maximum_run_hours:
        return TrainingBudget(20, 3, three_epochs, "20h_3epochs_within_budget")
    two_epochs = project(optimizer_steps_per_epoch_20h, 2)
    if two_epochs <= maximum_run_hours:
        return TrainingBudget(20, 2, two_epochs, "reduced_to_2epochs")
    ten_hour_two_epochs = project(optimizer_steps_per_epoch_20h / 2.0, 2)
    return TrainingBudget(
        10,
        2,
        ten_hour_two_epochs,
        "reduced_both_formal_runs_to_10h",
    )

