"""Cross-file validation for the frozen no-data experiment protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from robust_asr.config import canonical_sha256, load_json_object, require_keys
from robust_asr.experiments import (
    FrontendCondition,
    ModelCondition,
    build_formal_reverb_matrix,
    total_asr_inputs,
)

CONFIG_FILES = (
    "data.json",
    "rir.json",
    "wpe.json",
    "whisper.json",
    "lora.json",
    "evaluation.json",
)


@dataclass(frozen=True)
class ProtocolSummary:
    protocol_sha256: str
    sample_rate: int
    microphone_count: int
    model_count: int
    frontend_count: int
    rt60_count: int
    formal_reverb_inputs: int
    missing_baseline_dependencies: tuple[str, ...]
    missing_training_dependencies: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "sample_rate": self.sample_rate,
            "microphone_count": self.microphone_count,
            "model_count": self.model_count,
            "frontend_count": self.frontend_count,
            "rt60_count": self.rt60_count,
            "formal_reverb_inputs": self.formal_reverb_inputs,
            "missing_baseline_dependencies": list(
                self.missing_baseline_dependencies
            ),
            "missing_training_dependencies": list(
                self.missing_training_dependencies
            ),
        }


def _positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _check_equal_sample_rates(configs: Mapping[str, Mapping[str, Any]]) -> int:
    rates = {
        name: value.get("sample_rate")
        for name, value in configs.items()
        if "sample_rate" in value
    }
    if not rates:
        raise ValueError("protocol has no sample_rate")
    unique = set(rates.values())
    if len(unique) != 1:
        raise ValueError(f"sample_rate mismatch across configs: {rates}")
    return _positive_int(next(iter(unique)), name="sample_rate")


def _missing_dependencies(names: tuple[str, ...]) -> tuple[str, ...]:
    import importlib.util

    return tuple(name for name in names if importlib.util.find_spec(name) is None)


def load_and_validate_protocol(config_directory: str | Path) -> ProtocolSummary:
    """Validate frozen JSON configs without touching speech data or models."""

    root = Path(config_directory)
    configs = {
        filename.removesuffix(".json"): load_json_object(root / filename)
        for filename in CONFIG_FILES
    }
    for name, value in configs.items():
        require_keys(value, {"schema_version"}, context=name)
        if value["schema_version"] != 1:
            raise ValueError(f"unsupported {name} schema_version")

    sample_rate = _check_equal_sample_rates(configs)
    data = configs["data"]
    rir = configs["rir"]
    wpe = configs["wpe"]
    whisper = configs["whisper"]
    lora = configs["lora"]
    evaluation = configs["evaluation"]

    require_keys(
        data,
        {"train_subset_hours", "dev_model_utterances", "dev_frontend_utterances"},
        context="data",
    )
    require_keys(
        rir,
        {"microphone_count", "reference_channel", "test_rt60_seconds"},
        context="rir",
    )
    require_keys(wpe, {"conditions", "formal_backend"}, context="wpe")
    require_keys(
        whisper,
        {
            "model_id",
            "revision",
            "model_safetensors_sha256",
            "language",
            "task",
        },
        context="whisper",
    )
    require_keys(
        lora,
        {
            "rank",
            "task_type",
            "peft_wrapper",
            "pilot_targets",
            "formal_runs",
            "dataloader_num_workers",
            "dataloader_prefetch_factor",
            "logging",
        },
        context="lora",
    )
    require_keys(
        evaluation,
        {"models", "frontends", "test_reverb_utterances", "rt60_seconds"},
        context="evaluation",
    )

    microphone_count = _positive_int(
        rir["microphone_count"], name="microphone_count"
    )
    if microphone_count != 4:
        raise ValueError("v0.1 requires four microphones")
    reference_channel = rir["reference_channel"]
    if reference_channel != 0:
        raise ValueError("v0.1 fixes reference_channel=0")
    if wpe["formal_backend"] != "nara_wpe":
        raise ValueError("formal WPE backend must be nara_wpe")
    if whisper["model_id"] != "openai/whisper-small":
        raise ValueError("v0.1 fixes openai/whisper-small")
    if not isinstance(whisper["revision"], str) or len(whisper["revision"]) != 40:
        raise ValueError("Whisper revision must be a pinned 40-character commit")
    if (
        not isinstance(whisper["model_safetensors_sha256"], str)
        or len(whisper["model_safetensors_sha256"]) != 64
    ):
        raise ValueError("Whisper model SHA-256 must be pinned")
    if whisper["language"] != "zh" or whisper["task"] != "transcribe":
        raise ValueError("Whisper must use zh transcription")
    if lora["rank"] != 8:
        raise ValueError("v0.1 fixes LoRA rank=8")
    if lora["task_type"] is not None:
        raise ValueError("Whisper LoRA must not use the text Seq2Seq PEFT wrapper")
    if lora["peft_wrapper"] != "generic_for_whisper_input_features":
        raise ValueError("Whisper LoRA must preserve input_features")
    train_hours = _positive_int(data["train_subset_hours"], name="train_subset_hours")
    fallback_hours = _positive_int(
        data["fallback_train_subset_hours"],
        name="fallback_train_subset_hours",
    )
    if fallback_hours >= train_hours:
        raise ValueError("fallback training subset must be smaller")
    clean_probability = float(data["mct_clean_probability"])
    reverb_probability = float(data["mct_raw_reverb_probability"])
    if abs(clean_probability + reverb_probability - 1.0) > 1e-12:
        raise ValueError("MCT clean and reverb probabilities must sum to one")
    if any(value < 0.0 or value > 1.0 for value in (clean_probability, reverb_probability)):
        raise ValueError("MCT probabilities must be in [0, 1]")
    if evaluation["test_reverb_utterances"] != data["test_reverb_utterances"]:
        raise ValueError("data and evaluation test utterance counts disagree")
    if lora["per_device_train_batch_size"] * lora["gradient_accumulation_steps"] != lora[
        "effective_batch_size"
    ]:
        raise ValueError("LoRA effective batch size is inconsistent")
    for field in ("dataloader_num_workers", "dataloader_prefetch_factor"):
        value = lora[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"LoRA {field} must be a positive integer")
    logging = lora["logging"]
    if not isinstance(logging, Mapping):
        raise ValueError("LoRA logging config must be an object")
    require_keys(
        logging,
        {
            "console_interval_steps",
            "structured_interval_steps",
            "selection_metric",
            "maximum_clean_cer_degradation_pp",
            "heavy_rt60_seconds",
        },
        context="lora.logging",
    )
    console_interval = _positive_int(
        logging["console_interval_steps"], name="console_interval_steps"
    )
    structured_interval = _positive_int(
        logging["structured_interval_steps"], name="structured_interval_steps"
    )
    if structured_interval > console_interval:
        raise ValueError("structured logs cannot be less frequent than console output")
    if logging["selection_metric"] != "dev_reverb_cer":
        raise ValueError("checkpoint selection metric must be dev_reverb_cer")
    maximum_clean_degradation = float(
        logging["maximum_clean_cer_degradation_pp"]
    )
    if maximum_clean_degradation < 0 or not math.isfinite(
        maximum_clean_degradation
    ):
        raise ValueError("maximum clean CER degradation must be finite and non-negative")
    expected_wpe_taps = {"s_wpe_10": 10, "s_wpe_40": 40, "m_wpe_10": 10}
    if wpe.get("taps") != expected_wpe_taps:
        raise ValueError("WPE tap conditions disagree with the frozen protocol")

    models = tuple(ModelCondition(value) for value in evaluation["models"])
    frontends = tuple(
        FrontendCondition(value) for value in evaluation["frontends"]
    )
    rt60 = tuple(float(value) for value in evaluation["rt60_seconds"])
    if tuple(float(value) for value in rir["test_rt60_seconds"]) != rt60:
        raise ValueError("RIR and evaluation RT60 grids disagree")
    heavy_rt60 = tuple(float(value) for value in logging["heavy_rt60_seconds"])
    if not heavy_rt60 or any(value not in rt60 for value in heavy_rt60):
        raise ValueError("heavy RT60 values must be a non-empty evaluation subset")
    if tuple(wpe["conditions"]) != tuple(value.value for value in frontends):
        raise ValueError("WPE and evaluation frontend order disagree")
    if set(models) != set(ModelCondition):
        raise ValueError("formal evaluation must include W0/W1/W2")
    if set(frontends) != set(FrontendCondition):
        raise ValueError("formal evaluation must include all four frontends")

    utterances = _positive_int(
        evaluation["test_reverb_utterances"],
        name="test_reverb_utterances",
    )
    matrix = build_formal_reverb_matrix(
        utterance_count=utterances,
        models=models,
        frontends=frontends,
        rt60_seconds=rt60,
    )
    protocol_sha = canonical_sha256(configs)
    return ProtocolSummary(
        protocol_sha256=protocol_sha,
        sample_rate=sample_rate,
        microphone_count=microphone_count,
        model_count=len(models),
        frontend_count=len(frontends),
        rt60_count=len(rt60),
        formal_reverb_inputs=total_asr_inputs(matrix),
        missing_baseline_dependencies=_missing_dependencies(
            (
                "pyroomacoustics",
                "nara_wpe",
                "torch",
                "transformers",
                "opencc",
            )
        ),
        missing_training_dependencies=_missing_dependencies(
            ("accelerate", "datasets", "peft")
        ),
    )
