"""Enhance a WAV through the persistent RNNoise C API streaming path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from speech_frontend.audio import AudioData, read_audio, write_audio
from speech_frontend.rnnoise import (
    RNNoiseLibrary,
    StreamingRNNoise16k,
    StreamingRNNoise48k,
)
from speech_frontend.rnnoise.controller import (
    CorrectionAwareConfig,
    correction_aware_mix,
    fixed_residual_mix,
)


def delay_compensate(samples: np.ndarray, delay_samples: int) -> np.ndarray:
    if delay_samples == 0:
        return samples.copy()
    if delay_samples >= samples.size:
        return np.zeros_like(samples)
    return np.pad(samples[delay_samples:], (0, delay_samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--compensate-delay", action="store_true")
    parser.add_argument("--pcm-compatible", action="store_true")
    parser.add_argument("--vad-json", type=Path)
    parser.add_argument(
        "--residual-strength",
        type=float,
        default=0.50,
        help=(
            "RNNoise weight for experimental conservative mixing; "
            "1.0 is identical to official R3"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("official", "conservative", "aggressive"),
        default="official",
    )
    arguments = parser.parse_args()

    audio = read_audio(arguments.input)
    if audio.sample_rate not in (16_000, 48_000):
        raise ValueError("RNNoise streaming input must be 16 kHz or 48 kHz")
    if arguments.chunk_size is not None and arguments.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if not 0.0 <= arguments.residual_strength <= 1.0:
        raise ValueError("residual-strength must be within [0, 1]")
    chunk_size = arguments.chunk_size or (
        160 if audio.sample_rate == 16_000 else 480
    )

    library = RNNoiseLibrary(arguments.library)
    if audio.sample_rate == 16_000:
        stream = StreamingRNNoise16k(
            library,
            pcm_compatible=arguments.pcm_compatible,
        )
    else:
        stream = StreamingRNNoise48k(
            library,
            pcm_compatible=arguments.pcm_compatible,
        )

    outputs: list[np.ndarray] = []
    vad: list[np.ndarray] = []
    started = time.perf_counter()
    for offset in range(0, audio.samples.size, chunk_size):
        result = stream.process_chunk(
            audio.samples[offset : offset + chunk_size]
        )
        outputs.append(result.samples)
        vad.append(result.vad_probabilities)
    result = stream.flush()
    outputs.append(result.samples)
    vad.append(result.vad_probabilities)
    elapsed = time.perf_counter() - started

    enhanced = np.concatenate(outputs)
    vad_probabilities = np.concatenate(vad)
    algorithmic_delay_samples = stream.algorithmic_delay_samples
    alignment_delay_samples = stream.alignment_delay_samples
    if arguments.compensate_delay:
        enhanced = delay_compensate(enhanced, alignment_delay_samples)
    if arguments.mode != "official" and not arguments.compensate_delay:
        raise ValueError(
            "conservative/aggressive file modes require --compensate-delay"
        )
    mean_strength = 1.0
    if arguments.mode == "conservative":
        enhanced = fixed_residual_mix(
            audio.samples,
            enhanced,
            arguments.residual_strength,
        )
        mean_strength = arguments.residual_strength
    elif arguments.mode == "aggressive":
        controlled = correction_aware_mix(
            audio.samples,
            enhanced,
            vad_probabilities=vad_probabilities,
            config=CorrectionAwareConfig(
                correction_threshold_db=-9.0,
                speech_protection=0.30,
            ),
        )
        enhanced = controlled.samples
        mean_strength = float(np.mean(controlled.frame_strength))
    write_audio(
        arguments.output,
        AudioData(enhanced.astype(np.float32), audio.sample_rate),
    )

    duration = audio.samples.size / audio.sample_rate
    method = {
        "official": "R3_official_rnnoise_c_api_streaming",
        "aggressive": "R4_aggressive_correction_-9db_vad030",
    }.get(
        arguments.mode,
        f"R4_experimental_fixed_{arguments.residual_strength:.2f}",
    )
    report = {
        "input": str(arguments.input),
        "output": str(arguments.output),
        "method": method,
        "mode": arguments.mode,
        "mean_enhancement_strength": mean_strength,
        "sample_rate": audio.sample_rate,
        "chunk_size": chunk_size,
        "input_samples": int(audio.samples.size),
        "output_samples": int(enhanced.size),
        "algorithmic_delay_samples": algorithmic_delay_samples,
        "algorithmic_delay_ms": (
            1_000 * algorithmic_delay_samples / audio.sample_rate
        ),
        "file_alignment_delay_samples": alignment_delay_samples,
        "file_alignment_delay_ms": (
            1_000 * alignment_delay_samples / audio.sample_rate
        ),
        "delay_compensated": arguments.compensate_delay,
        "vad_frames": int(vad_probabilities.size),
        "vad_mean": float(np.mean(vad_probabilities)),
        "rtf_processing_only": elapsed / duration,
        "peak": float(np.max(np.abs(enhanced), initial=0.0)),
        "clipping_samples": int(np.count_nonzero(np.abs(enhanced) > 1.0)),
        "resampler_input_clipping_samples": int(
            getattr(stream, "resampler_clipping_samples", 0)
        ),
    }
    if arguments.vad_json is not None:
        arguments.vad_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.vad_json.write_text(
            json.dumps(
                {
                    **report,
                    "vad_probability": vad_probabilities.tolist(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
