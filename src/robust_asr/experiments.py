"""Frozen experiment identifiers and formal factorial matrix generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


class ModelCondition(str, Enum):
    PRETRAINED = "w0_pretrained"
    CLEAN_LORA = "w1_clean_lora"
    MCT_LORA = "w2_mct_lora"


class FrontendCondition(str, Enum):
    RAW = "raw"
    S_WPE_10 = "s_wpe_10"
    S_WPE_40 = "s_wpe_40"
    M_WPE_10 = "m_wpe_10"


FORMAL_RT60_SECONDS = (0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class ExperimentCell:
    experiment_id: str
    model: ModelCondition
    frontend: FrontendCondition
    rt60_seconds: float
    utterance_count: int

    def as_dict(self) -> dict[str, str | float | int]:
        value = asdict(self)
        value["model"] = self.model.value
        value["frontend"] = self.frontend.value
        return value


def _rt60_token(value: float) -> str:
    return f"rt{round(value * 100):03d}"


def build_formal_reverb_matrix(
    *,
    utterance_count: int = 1_000,
    models: Iterable[ModelCondition] = tuple(ModelCondition),
    frontends: Iterable[FrontendCondition] = tuple(FrontendCondition),
    rt60_seconds: Iterable[float] = FORMAL_RT60_SECONDS,
) -> tuple[ExperimentCell, ...]:
    """Build the 3 model × 4 frontend × 5 RT60 formal matrix."""

    if utterance_count <= 0:
        raise ValueError("utterance_count must be positive")
    model_values = tuple(models)
    frontend_values = tuple(frontends)
    rt60_values = tuple(rt60_seconds)
    if len(set(model_values)) != len(model_values) or not model_values:
        raise ValueError("models must be non-empty and unique")
    if len(set(frontend_values)) != len(frontend_values) or not frontend_values:
        raise ValueError("frontends must be non-empty and unique")
    if len(set(rt60_values)) != len(rt60_values) or not rt60_values:
        raise ValueError("rt60_seconds must be non-empty and unique")
    if any(value <= 0 for value in rt60_values):
        raise ValueError("RT60 values must be positive")

    cells: list[ExperimentCell] = []
    for model in model_values:
        for frontend in frontend_values:
            for rt60 in rt60_values:
                identifier = "__".join(
                    [model.value, frontend.value, _rt60_token(rt60)]
                )
                cells.append(
                    ExperimentCell(
                        experiment_id=identifier,
                        model=model,
                        frontend=frontend,
                        rt60_seconds=float(rt60),
                        utterance_count=utterance_count,
                    )
                )
    return tuple(cells)


def total_asr_inputs(cells: Iterable[ExperimentCell]) -> int:
    return sum(cell.utterance_count for cell in cells)


def wpe_lora_interaction(
    *,
    pretrained_raw_cer: float,
    pretrained_m_wpe_cer: float,
    mct_raw_cer: float,
    mct_m_wpe_cer: float,
) -> float:
    """Return ΔWPE(MCT) - ΔWPE(pretrained); negative means synergy."""

    values = (
        pretrained_raw_cer,
        pretrained_m_wpe_cer,
        mct_raw_cer,
        mct_m_wpe_cer,
    )
    if any(value < 0 for value in values):
        raise ValueError("CER values cannot be negative")
    pretrained_delta = pretrained_m_wpe_cer - pretrained_raw_cer
    mct_delta = mct_m_wpe_cer - mct_raw_cer
    return mct_delta - pretrained_delta

