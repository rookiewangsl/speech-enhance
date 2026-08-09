"""Create one compact diagnostic figure for an enhancement experiment."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

_CACHE_DIRECTORY = Path(tempfile.gettempdir()) / "speech_frontend_matplotlib"
_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(_CACHE_DIRECTORY),
)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIRECTORY))

import matplotlib.pyplot as plt
import numpy as np

from speech_frontend.audio import read_audio
from speech_frontend.enhancement.wiener import WienerConfig
from speech_frontend.pipeline import ClassicalEnhancer
from speech_frontend.vad.statistical import StatisticalVAD, StatisticalVADConfig


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def log_spectrogram(
    samples: np.ndarray,
    *,
    sample_rate: int,
    frame_length: int = 512,
    hop_length: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = max(1, int(np.ceil((samples.size - frame_length) / hop_length)) + 1)
    padded_length = (frame_count - 1) * hop_length + frame_length
    padded = np.pad(samples, (0, max(0, padded_length - samples.size)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length]
    window = np.hanning(frame_length + 1)[:-1]
    spectrum = np.fft.rfft(frames * window, axis=1)
    db = 10.0 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    time = np.arange(frame_count) * hop_length / sample_rate
    frequency = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)
    return time, frequency, db.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--method",
        default="mcra_dd_wiener",
        choices=(
            "mcra_dd_wiener",
            "mcra_om_lsa",
            "imcra_om_lsa",
            "dual_uncertainty_wiener",
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/plots/diagnostic.png"))
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()
    rows = [
        json.loads(line)
        for line in resolve(project_root, str(arguments.manifest)).read_text().splitlines()
    ]
    row = rows[arguments.index]
    clean = read_audio(resolve(project_root, row["clean"]))
    noisy = read_audio(resolve(project_root, row["noisy"]))
    statistical_config = StatisticalVADConfig(
        alpha_dd=0.96,
        score_threshold_on=20.0,
        score_threshold_off=10.0,
    )
    enhancer = ClassicalEnhancer(
        wiener_config=WienerConfig(alpha_dd=0.92, gain_floor=0.30),
        statistical_vad_config=statistical_config,
    )
    result = enhancer.enhance(noisy.samples, method=arguments.method)
    vad = StatisticalVAD(statistical_config).detect(noisy.samples)

    figure, axes = plt.subplots(5, 1, figsize=(12, 14), constrained_layout=True)
    for axis, samples, title in zip(
        axes[:3],
        (clean.samples, noisy.samples, result.samples),
        ("Clean reference", "Noisy input", f"Enhanced: {arguments.method}"),
        strict=True,
    ):
        time, frequency, db = log_spectrogram(samples, sample_rate=clean.sample_rate)
        image = axis.pcolormesh(time, frequency, db, shading="auto", cmap="magma")
        axis.set(title=title, ylabel="Hz", ylim=(0, 8_000))
        figure.colorbar(image, ax=axis, label="dB")

    enhancement_time = (
        np.arange(result.diagnostics.noise_psd.shape[0])
        * enhancer.stft.config.hop_length
        / clean.sample_rate
    )
    axes[3].plot(
        enhancement_time,
        10.0 * np.log10(np.mean(result.diagnostics.noise_psd, axis=1) + 1e-12),
        label="mean MCRA noise PSD",
    )
    axes[3].plot(
        enhancement_time,
        np.mean(result.diagnostics.gain, axis=1),
        label="mean gain",
    )
    axes[3].set(title="Noise tracking and gain", ylabel="dB / gain")
    axes[3].legend(loc="best")

    vad_time = np.arange(vad.speech_probability.size) * 0.01
    axes[4].plot(vad_time, vad.speech_probability, label="statistical VAD probability")
    axes[4].step(vad_time, vad.frame_labels.astype(float), where="post", label="state-machine label")
    axes[4].set(title="VAD trace", xlabel="Time (s)", ylabel="probability", ylim=(-0.05, 1.05))
    axes[4].legend(loc="best")

    output = resolve(project_root, str(arguments.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
