from __future__ import annotations

import pytest

from robust_asr.experiments import (
    build_formal_reverb_matrix,
    total_asr_inputs,
    wpe_lora_interaction,
)
from robust_asr.lora import (
    LoRAProtocol,
    LoRATarget,
    choose_training_budget,
    select_lora_target_names,
    validate_whisper_target_count,
)


def whisper_module_names() -> list[str]:
    names: list[str] = []
    for layer in range(12):
        encoder = f"model.model.encoder.layers.{layer}.self_attn"
        decoder_self = f"model.model.decoder.layers.{layer}.self_attn"
        decoder_cross = f"model.model.decoder.layers.{layer}.encoder_attn"
        for prefix in (encoder, decoder_self, decoder_cross):
            names.extend(
                [
                    f"{prefix}.q_proj",
                    f"{prefix}.k_proj",
                    f"{prefix}.v_proj",
                    f"{prefix}.out_proj",
                ]
            )
    return names


def test_lora_target_selection_matches_expected_whisper_small_counts() -> None:
    names = whisper_module_names()
    encoder = select_lora_target_names(names, LoRATarget.ENCODER_QV)
    full = select_lora_target_names(
        names, LoRATarget.ENCODER_DECODER_QV
    )

    validate_whisper_target_count(encoder, LoRATarget.ENCODER_QV)
    validate_whisper_target_count(full, LoRATarget.ENCODER_DECODER_QV)
    assert len(encoder) == 24
    assert len(full) == 72
    assert all(name.endswith((".q_proj", ".v_proj")) for name in full)


def test_lora_protocol_has_frozen_scaling() -> None:
    assert LoRAProtocol().scaling == 2.0


@pytest.mark.parametrize(
    ("seconds_per_step", "expected_hours", "expected_epochs"),
    [(1.0, 20, 3), (5.0, 20, 2), (20.0, 10, 2)],
)
def test_training_budget_follows_frozen_fallback(
    seconds_per_step: float,
    expected_hours: int,
    expected_epochs: int,
) -> None:
    budget = choose_training_budget(
        seconds_per_optimizer_step=seconds_per_step,
        optimizer_steps_per_epoch_20h=2_000,
    )

    assert budget.subset_hours == expected_hours
    assert budget.epochs == expected_epochs


def test_formal_matrix_has_60_cells_and_60k_inputs() -> None:
    matrix = build_formal_reverb_matrix()

    assert len(matrix) == 60
    assert len({cell.experiment_id for cell in matrix}) == 60
    assert total_asr_inputs(matrix) == 60_000


def test_interaction_sign_convention() -> None:
    interaction = wpe_lora_interaction(
        pretrained_raw_cer=0.20,
        pretrained_m_wpe_cer=0.18,
        mct_raw_cer=0.12,
        mct_m_wpe_cer=0.08,
    )

    assert interaction == pytest.approx(-0.02)

