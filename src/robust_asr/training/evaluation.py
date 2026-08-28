"""Whisper LoRA dev decoding on clean and controlled raw reverberation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from robust_asr.baseline import run_frozen_baseline
from robust_asr.manifest import read_jsonl
from robust_asr.training.engine import EpochEvaluation
from robust_asr.training.selection import DevCheckpointMetrics


def trainable_parameter_sha256(model: Any) -> str:
    """Hash LoRA tensors so cached dev predictions cannot cross checkpoints."""

    digest = hashlib.sha256()
    found = 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        values = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
        found += 1
    if found == 0:
        raise ValueError("model has no trainable parameters to fingerprint")
    return digest.hexdigest()


class LoadedWhisperTranscriber:
    """Use an already loaded base/PEFT Whisper model for deterministic decoding."""

    def __init__(
        self,
        *,
        processor: Any,
        model: Any,
        model_id: str,
        base_revision: str,
        adapter_sha256: str,
        device: str = "cuda",
        num_beams: int = 1,
    ) -> None:
        if not adapter_sha256 or len(adapter_sha256) != 64:
            raise ValueError("adapter_sha256 must be a 64-character digest")
        if (
            not isinstance(num_beams, int)
            or isinstance(num_beams, bool)
            or num_beams <= 0
        ):
            raise ValueError("num_beams must be a positive integer")
        self.processor = processor
        self.model = model
        self.model_id = model_id
        self.model_revision = f"{base_revision}+lora:{adapter_sha256}"
        self.device = device
        self.num_beams = num_beams

    def transcribe(self, audio: np.ndarray, *, sample_rate: int = 16_000) -> str:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional training stack
            raise RuntimeError("Whisper dev evaluation requires torch") from exc
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Whisper input must be non-empty mono audio")
        if sample_rate != 16_000:
            raise ValueError("Whisper dev evaluation requires 16 kHz audio")
        if waveform.size > 30 * sample_rate:
            raise ValueError("Whisper dev input exceeds 30 seconds")
        if not np.all(np.isfinite(waveform)):
            raise ValueError("Whisper dev input contains NaN or infinity")
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        generation: dict[str, Any] = {
            "input_features": inputs.input_features.to(self.device),
            "language": "zh",
            "task": "transcribe",
            "do_sample": False,
            "num_beams": self.num_beams,
            "max_length": 225,
        }
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            generation["attention_mask"] = attention_mask.to(self.device)
        self.model.eval()
        with torch.inference_mode():
            token_ids = self.model.generate(**generation)
        return str(
            self.processor.batch_decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        ).strip()


class WhisperDevEvaluator:
    """Evaluate one LoRA epoch without reading the sealed test split."""

    def __init__(
        self,
        *,
        processor: Any,
        manifest_path: str | Path,
        corpus_root: str | Path,
        rir_manifest_path: str | Path,
        rir_root: str | Path,
        output_dir: str | Path,
        model_id: str,
        base_revision: str,
        limit: int = 1_000,
        rt60_seconds: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
        robust_rt60_seconds: Sequence[float] = (0.4, 0.6, 0.8, 1.0),
        heavy_rt60_seconds: Sequence[float] = (0.8, 1.0),
        device: str = "cuda",
        num_beams: int = 1,
        seed: int = 2026,
        bootstrap_draws: int = 1_000,
    ) -> None:
        if limit <= 0 or bootstrap_draws <= 0:
            raise ValueError("evaluation limit and bootstrap draws must be positive")
        self.processor = processor
        self.manifest_path = Path(manifest_path)
        self.corpus_root = Path(corpus_root)
        self.rir_manifest_path = Path(rir_manifest_path)
        self.rir_root = Path(rir_root)
        self.output_dir = Path(output_dir)
        self.model_id = model_id
        self.base_revision = base_revision
        self.limit = limit
        self.rt60_seconds = tuple(map(float, rt60_seconds))
        self.robust_rt60_seconds = tuple(map(float, robust_rt60_seconds))
        self.heavy_rt60_seconds = tuple(map(float, heavy_rt60_seconds))
        self.device = device
        self.num_beams = num_beams
        self.seed = seed
        self.bootstrap_draws = bootstrap_draws
        if not self.rt60_seconds or len(set(self.rt60_seconds)) != len(
            self.rt60_seconds
        ):
            raise ValueError("evaluation RT60 values must be non-empty and unique")
        grid = set(self.rt60_seconds)
        if not set(self.robust_rt60_seconds) <= grid:
            raise ValueError("robust RT60 values must belong to the evaluation grid")
        if not set(self.heavy_rt60_seconds) <= grid:
            raise ValueError("heavy RT60 values must belong to the evaluation grid")

    def __call__(self, model: Any, epoch: int) -> EpochEvaluation:
        adapter_sha = trainable_parameter_sha256(model)
        transcriber = LoadedWhisperTranscriber(
            processor=self.processor,
            model=model,
            model_id=self.model_id,
            base_revision=self.base_revision,
            adapter_sha256=adapter_sha,
            device=self.device,
            num_beams=self.num_beams,
        )
        epoch_dir = self.output_dir / "dev" / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        output_path = epoch_dir / "predictions.jsonl"
        summary = run_frozen_baseline(
            manifest_path=self.manifest_path,
            corpus_root=self.corpus_root,
            rir_manifest_path=self.rir_manifest_path,
            rir_root=self.rir_root,
            output_path=output_path,
            transcriber=transcriber,
            limit=self.limit,
            frontends=("raw",),
            rt60_seconds=self.rt60_seconds,
            seed=self.seed,
            bootstrap_draws=self.bootstrap_draws,
            checkpoint_every_results=20,
        )
        conditions = summary["conditions"]
        clean_rows = [
            row
            for row in conditions
            if row["frontend"] == "clean"
            and row["target_rt60_seconds"] is None
        ]
        raw_rows = {
            float(row["target_rt60_seconds"]): row
            for row in conditions
            if row["frontend"] == "raw"
            and row["target_rt60_seconds"] is not None
        }
        if len(clean_rows) != 1 or set(raw_rows) != set(self.rt60_seconds):
            raise ValueError("dev baseline summary lacks a required condition")
        per_rt60 = {rt60: float(raw_rows[rt60]["cer"]) for rt60 in self.rt60_seconds}
        robust = float(np.mean([per_rt60[value] for value in self.robust_rt60_seconds]))
        heavy = float(np.mean([per_rt60[value] for value in self.heavy_rt60_seconds]))
        robust_rows = [
            row
            for row in read_jsonl(output_path)
            if row.get("frontend") == "raw"
            and row.get("target_rt60_seconds") in self.robust_rt60_seconds
        ]
        return EpochEvaluation(
            metrics=DevCheckpointMetrics(
                epoch=epoch,
                clean_cer=float(clean_rows[0]["cer"]),
                reverb_cer=robust,
                heavy_cer=heavy,
            ),
            per_rt60_cer=per_rt60,
            predictions=read_jsonl(output_path),
            substitutions=sum(int(row["substitutions"]) for row in robust_rows),
            deletions=sum(int(row["deletions"]) for row in robust_rows),
            insertions=sum(int(row["insertions"]) for row in robust_rows),
        )
