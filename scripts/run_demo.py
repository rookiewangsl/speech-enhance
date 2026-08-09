"""Generate a self-contained RNNoise comparison demo from one WAV file."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from speech_frontend.audio import AudioData, read_audio, write_audio
from speech_frontend.metrics import si_sdr, stoi
from speech_frontend.rnnoise import (
    ContinuityConfig,
    RNNoiseLibrary,
    StreamingRNNoise16k,
    StreamingRNNoise48k,
    stabilize_output_continuity,
)


def delay_compensate(samples: np.ndarray, delay: int) -> np.ndarray:
    if delay <= 0:
        return samples.copy()
    if delay >= samples.size:
        return np.zeros_like(samples)
    return np.pad(samples[delay:], (0, delay))


def run_stream(
    samples: np.ndarray,
    sample_rate: int,
    library: RNNoiseLibrary,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    if sample_rate == 16_000:
        stream = StreamingRNNoise16k(library)
    elif sample_rate == 48_000:
        stream = StreamingRNNoise48k(library)
    else:
        raise ValueError("demo input must be 16 kHz or 48 kHz")
    output: list[np.ndarray] = []
    vad: list[np.ndarray] = []
    started = time.perf_counter()
    for offset in range(0, samples.size, chunk_size):
        result = stream.process_chunk(samples[offset : offset + chunk_size])
        output.append(result.samples)
        vad.append(result.vad_probabilities)
    result = stream.flush()
    output.append(result.samples)
    vad.append(result.vad_probabilities)
    elapsed = time.perf_counter() - started
    enhanced = delay_compensate(
        np.concatenate(output),
        stream.alignment_delay_samples,
    )
    probabilities = np.concatenate(vad)
    return (
        enhanced,
        probabilities,
        stream.algorithmic_delay_samples,
        elapsed,
    )


def spectrogram(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = sample_rate // 50
    hop = sample_rate // 100
    transform = ShortTimeFFT(
        hann(frame, sym=False),
        hop=hop,
        fs=sample_rate,
        mfft=2 ** int(np.ceil(np.log2(frame))),
        scale_to="magnitude",
    )
    magnitude = np.abs(transform.stft(samples))
    return 20.0 * np.log10(np.maximum(magnitude, 1e-6))


def save_plot(
    noisy: np.ndarray,
    r3: np.ndarray,
    continuity: np.ndarray,
    vad: np.ndarray,
    sample_rate: int,
    output: Path,
) -> None:
    duration = noisy.size / sample_rate
    waveform_time = np.arange(noisy.size) / sample_rate
    frame_time = np.arange(vad.size) * 0.010
    noisy_spectrum = spectrogram(noisy, sample_rate)
    r3_spectrum = spectrogram(r3, sample_rate)
    extent = [0.0, duration, 0.0, sample_rate / 2]

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(12, 9),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(waveform_time, noisy, linewidth=0.45, label="noisy")
    axes[0].plot(waveform_time, r3, linewidth=0.45, label="R3")
    axes[0].plot(
        waveform_time,
        continuity,
        linewidth=0.45,
        label="C1 continuity",
        alpha=0.8,
    )
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", ncols=3)
    axes[0].grid(alpha=0.2)
    axes[1].imshow(
        noisy_spectrum,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-75,
        vmax=-10,
        cmap="magma",
    )
    axes[1].set_ylabel("Noisy Hz")
    axes[2].imshow(
        r3_spectrum,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-75,
        vmax=-10,
        cmap="magma",
    )
    axes[2].set_ylabel("R3 Hz")
    axes[3].step(frame_time, vad, where="post")
    axes[3].axhline(0.60, color="black", linestyle="--", linewidth=0.8)
    axes[3].set_ylim(-0.02, 1.02)
    axes[3].set_ylabel("VAD")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--chunk-size", type=int)
    arguments = parser.parse_args()

    noisy = read_audio(arguments.input)
    chunk_size = arguments.chunk_size or noisy.sample_rate // 100
    if chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    library = RNNoiseLibrary(arguments.library)
    r3, vad, delay_samples, elapsed = run_stream(
        noisy.samples,
        noisy.sample_rate,
        library,
        chunk_size,
    )
    frame_samples = noisy.sample_rate // 100
    continuity_result = stabilize_output_continuity(
        r3,
        vad,
        config=ContinuityConfig(
            sample_rate=noisy.sample_rate,
            frame_samples=frame_samples,
            max_boost_db=3.0,
        ),
    )
    output_root = arguments.output_dir
    write_audio(output_root / "noisy_input.wav", noisy)
    write_audio(
        output_root / "r3_official.wav",
        AudioData(r3.astype(np.float32), noisy.sample_rate),
    )
    write_audio(
        output_root / "c1_continuity_probe.wav",
        AudioData(continuity_result.samples, noisy.sample_rate),
    )

    report: dict[str, object] = {
        "input": str(arguments.input),
        "sample_rate": noisy.sample_rate,
        "samples": int(noisy.samples.size),
        "duration_seconds": noisy.samples.size / noisy.sample_rate,
        "chunk_size": chunk_size,
        "algorithmic_delay_samples": delay_samples,
        "algorithmic_delay_ms": 1_000 * delay_samples / noisy.sample_rate,
        "processing_rtf": elapsed / (noisy.samples.size / noisy.sample_rate),
        "vad_frames": int(vad.size),
        "vad_mean": float(np.mean(vad)),
        "continuity_boosted_frame_fraction": float(
            np.mean(continuity_result.frame_gain_db > 1e-4)
        ),
        "continuity_mean_boost_db": float(
            np.mean(continuity_result.frame_gain_db)
        ),
        "continuity_maximum_boost_db": float(
            np.max(continuity_result.frame_gain_db, initial=0.0)
        ),
    }
    if arguments.clean is not None:
        clean = read_audio(arguments.clean)
        if clean.sample_rate != noisy.sample_rate:
            raise ValueError("clean/noisy sample rates do not match")
        if clean.samples.shape != noisy.samples.shape:
            raise ValueError("clean/noisy lengths do not match")
        write_audio(output_root / "clean_reference.wav", clean)
        input_si_sdr = si_sdr(clean.samples, noisy.samples)
        input_stoi = stoi(
            clean.samples,
            noisy.samples,
            sample_rate=noisy.sample_rate,
        )
        report["metrics"] = {
            "r3_si_sdri_db": si_sdr(clean.samples, r3) - input_si_sdr,
            "r3_stoi_improvement": (
                stoi(clean.samples, r3, sample_rate=noisy.sample_rate)
                - input_stoi
            ),
            "c1_si_sdri_db": (
                si_sdr(clean.samples, continuity_result.samples)
                - input_si_sdr
            ),
            "c1_stoi_improvement": (
                stoi(
                    clean.samples,
                    continuity_result.samples,
                    sample_rate=noisy.sample_rate,
                )
                - input_stoi
            ),
        }
    save_plot(
        noisy.samples,
        r3,
        continuity_result.samples,
        vad,
        noisy.sample_rate,
        output_root / "comparison.png",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
