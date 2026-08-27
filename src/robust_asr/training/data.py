"""Deterministic clean and multi-condition data for Whisper adaptation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

from robust_asr.acoustics.rir import convolve_multichannel
from robust_asr.config import canonical_sha256
from robust_asr.download import sha256_file
from robust_asr.manifest import (
    choose_mct_condition,
    choose_rir_id,
    read_jsonl,
)

TrainingMode = Literal["clean", "mct"]


def _inside(root: Path, relative: object, *, field: str) -> Path:
    value = Path(str(relative))
    if value.is_absolute():
        raise ValueError(f"{field} must be relative to its configured root")
    target = (root / value).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes its configured root: {value}") from exc
    return target


def _read_mono_16k(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    values = np.asarray(waveform, dtype=np.float32)
    if sample_rate != 16_000 or values.ndim != 1 or values.size == 0:
        raise ValueError(f"expected non-empty mono 16 kHz audio: {path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"audio contains NaN or infinite values: {path}")
    return values


class WhisperAdaptationDataset:
    """Load AISHELL lazily and apply an epoch-deterministic train RIR.

    The object intentionally remains independent of ``torch`` so that manifest
    and augmentation tests can run in the lightweight DSP environment.  When
    workers are used, the DataLoader must keep ``persistent_workers=False`` and
    a fresh iterator must be created after every :meth:`set_epoch` call.
    """

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        corpus_root: str | Path,
        mode: TrainingMode,
        rir_manifest_path: str | Path | None = None,
        rir_root: str | Path | None = None,
        reverb_probability: float = 0.5,
        seed: int = 2026,
        reference_channel: int = 0,
        target_rms_dbfs: float = -25.0,
        peak_headroom_db: float = 1.0,
        maximum_input_seconds: float = 30.0,
        verify_rir_sha256: bool = True,
    ) -> None:
        if mode not in {"clean", "mct"}:
            raise ValueError(f"unsupported training mode: {mode}")
        if not 0.0 <= reverb_probability <= 1.0:
            raise ValueError("reverb_probability must be in [0, 1]")
        if not 0 <= reference_channel < 4:
            raise ValueError("reference_channel must select one of four microphones")
        if maximum_input_seconds <= 0:
            raise ValueError("maximum_input_seconds must be positive")

        self.manifest_path = Path(manifest_path).resolve()
        self.corpus_root = Path(corpus_root).resolve()
        self.mode = mode
        self.reverb_probability = reverb_probability
        self.seed = seed
        self.reference_channel = reference_channel
        self.target_rms_dbfs = target_rms_dbfs
        self.peak_headroom_db = peak_headroom_db
        self.maximum_samples = round(maximum_input_seconds * 16_000)
        self.verify_rir_sha256 = verify_rir_sha256
        self.epoch = 0

        self.rows = read_jsonl(self.manifest_path)
        if not self.rows:
            raise ValueError("training manifest is empty")
        utterance_ids = [str(row.get("utterance_id", "")) for row in self.rows]
        if any(not value for value in utterance_ids):
            raise ValueError("training manifest contains an empty utterance_id")
        if len(utterance_ids) != len(set(utterance_ids)):
            raise ValueError("training manifest contains duplicate utterance_id values")
        for row in self.rows:
            if not str(row.get("audio_path", "")) or not str(row.get("transcript", "")):
                raise ValueError("training manifest requires audio_path and transcript")

        self.rir_root = None if rir_root is None else Path(rir_root).resolve()
        self.rir_manifest_path = (
            None if rir_manifest_path is None else Path(rir_manifest_path).resolve()
        )
        self.rirs: dict[str, dict[str, Any]] = {}
        if mode == "mct":
            if self.rir_root is None or self.rir_manifest_path is None:
                raise ValueError("MCT mode requires a train RIR manifest and root")
            rir_rows = read_jsonl(self.rir_manifest_path)
            for source in rir_rows:
                row = dict(source)
                rir_id = str(row.get("rir_id", ""))
                if not rir_id or rir_id in self.rirs:
                    raise ValueError("train RIR manifest has empty or duplicate rir_id")
                if row.get("split") != "train":
                    raise ValueError("MCT may only sample from the train RIR split")
                self.rirs[rir_id] = row
            if not self.rirs:
                raise ValueError("train RIR manifest is empty")

        self._rir_ids = tuple(sorted(self.rirs))
        self._verified_rir_paths: set[Path] = set()
        self.manifest_sha256 = canonical_sha256(self.rows)
        self.rir_manifest_sha256 = (
            None if not self.rirs else canonical_sha256(list(self.rirs.values()))
        )

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def decision(self, index: int) -> tuple[str, str | None]:
        row = self.rows[index]
        utterance_id = str(row["utterance_id"])
        if self.mode == "clean":
            return "clean", None
        condition = choose_mct_condition(
            utterance_id,
            epoch=self.epoch,
            seed=self.seed,
            reverb_probability=self.reverb_probability,
        )
        if condition == "clean":
            return condition, None
        return condition, choose_rir_id(
            utterance_id,
            self._rir_ids,
            epoch=self.epoch,
            seed=self.seed,
        )

    def _load_rir(self, rir_id: str) -> np.ndarray:
        row = self.rirs[rir_id]
        assert self.rir_root is not None
        path = _inside(self.rir_root, row.get("path"), field="RIR path")
        if self.verify_rir_sha256 and path not in self._verified_rir_paths:
            expected = row.get("file_sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"RIR manifest has no valid SHA-256: {rir_id}")
            observed = sha256_file(path)
            if observed != expected:
                raise ValueError(f"RIR SHA-256 mismatch: {rir_id}")
            self._verified_rir_paths.add(path)
        with np.load(path) as archive:
            full = np.asarray(archive["full"], dtype=np.float64)
        expected_shape = tuple(row.get("full_shape", ()))
        if full.ndim != 2 or full.shape[0] != 4:
            raise ValueError(f"RIR must have four channels: {rir_id} {full.shape}")
        if expected_shape and full.shape != expected_shape:
            raise ValueError(
                f"RIR shape disagrees with manifest: {rir_id} "
                f"{full.shape} != {expected_shape}"
            )
        if not np.all(np.isfinite(full)):
            raise ValueError(f"RIR contains NaN or infinite values: {rir_id}")
        return full

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio_path = _inside(
            self.corpus_root, row["audio_path"], field="AISHELL audio_path"
        )
        clean = _read_mono_16k(audio_path)
        condition, rir_id = self.decision(index)
        if rir_id is None:
            filters = np.asarray([[1.0]], dtype=np.float64)
            reference_channel = 0
            rir = None
        else:
            filters = self._load_rir(rir_id)
            reference_channel = self.reference_channel
            rir = self.rirs[rir_id]
        result = convolve_multichannel(
            clean,
            filters,
            reference_channel=reference_channel,
            target_rms_dbfs=self.target_rms_dbfs,
            peak_headroom_db=self.peak_headroom_db,
        )
        waveform = np.asarray(result.signals[reference_channel], dtype=np.float32)
        if waveform.size > self.maximum_samples:
            raise ValueError(
                f"Whisper input exceeds the configured limit after convolution: "
                f"{row['utterance_id']} ({waveform.size / 16_000:.3f} s)"
            )
        return {
            "utterance_id": str(row["utterance_id"]),
            "speaker_id": str(row.get("speaker_id", "")),
            "waveform": waveform,
            "sample_rate": 16_000,
            "transcript": str(row["transcript"]),
            "condition": condition,
            "rir_id": rir_id,
            "room_id": None if rir is None else rir.get("room_id"),
            "target_rt60_seconds": (
                None if rir is None else float(rir["target_rt60_seconds"])
            ),
            "reference_rms_dbfs": result.reference_rms_dbfs,
        }

    def audit(self) -> dict[str, Any]:
        """Return immutable run-identity fields for ``data_audit.json``."""

        return {
            "schema_version": 1,
            "mode": self.mode,
            "seed": self.seed,
            "epoch_dependent_sampling": self.mode == "mct",
            "utterances_per_epoch": len(self.rows),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "rir_manifest_path": (
                None
                if self.rir_manifest_path is None
                else str(self.rir_manifest_path)
            ),
            "rir_manifest_sha256": self.rir_manifest_sha256,
            "rir_count": len(self.rirs),
            "reverb_probability": (
                0.0 if self.mode == "clean" else self.reverb_probability
            ),
            "reference_channel": self.reference_channel,
            "target_rms_dbfs": self.target_rms_dbfs,
            "peak_headroom_db": self.peak_headroom_db,
            "maximum_input_samples": self.maximum_samples,
        }


class WhisperBatchCollator:
    """Convert variable-length waveforms and transcripts into a Whisper batch."""

    def __init__(
        self,
        processor: Any,
        *,
        decoder_start_token_id: int | None = None,
        sample_rate: int = 16_000,
    ) -> None:
        if sample_rate != 16_000:
            raise ValueError("Whisper training is frozen to 16 kHz")
        self.processor = processor
        self.sample_rate = sample_rate
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("cannot collate an empty batch")
        waveforms: list[np.ndarray] = []
        transcripts: list[str] = []
        for row in rows:
            waveform = np.asarray(row["waveform"], dtype=np.float32)
            if waveform.ndim != 1 or waveform.size == 0:
                raise ValueError("collator requires non-empty mono waveforms")
            if int(row.get("sample_rate", self.sample_rate)) != self.sample_rate:
                raise ValueError("collator received an unexpected sample rate")
            if not np.all(np.isfinite(waveform)):
                raise ValueError("collator received NaN or infinite audio")
            transcript = str(row.get("transcript", ""))
            if not transcript:
                raise ValueError("collator received an empty transcript")
            waveforms.append(waveform)
            transcripts.append(transcript)

        acoustic = self.processor.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=False,
        )
        tokens = self.processor.tokenizer(
            transcripts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        labels = tokens["input_ids"]
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask.ne(1), -100)
        decoder_start = self.decoder_start_token_id
        if decoder_start is None:
            decoder_start = getattr(self.processor.tokenizer, "bos_token_id", None)
        if (
            decoder_start is not None
            and labels.shape[1] > 0
            and bool((labels[:, 0] == decoder_start).all().item())
        ):
            labels = labels[:, 1:]

        batch = {
            "input_features": acoustic["input_features"],
            "labels": labels,
            "metadata": [
                {
                    key: row.get(key)
                    for key in (
                        "utterance_id",
                        "speaker_id",
                        "condition",
                        "rir_id",
                        "room_id",
                        "target_rt60_seconds",
                    )
                }
                for row in rows
            ],
        }
        return batch
