"""Visualize the MCRA noise-PSD estimate on the same STFT grid as its input."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

_CACHE_DIRECTORY = Path(tempfile.gettempdir()) / "speech_frontend_matplotlib"
_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIRECTORY))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIRECTORY))

import matplotlib.pyplot as plt
import numpy as np

from speech_frontend.audio import read_audio
from speech_frontend.enhancement.wiener import WienerConfig
from speech_frontend.pipeline import ClassicalEnhancer


def resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def read_manifest_row(manifest: Path, utterance_id: str) -> dict[str, object]:
    for line in manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("id") == utterance_id:
            return row
    raise ValueError(f"utterance id not found in manifest: {utterance_id}")


def power_to_db(values: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, floor))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot noisy STFT power, MCRA noise PSD, and DD-Wiener gain."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--id", required=True, dest="utterance_id")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    project_root = arguments.project_root.resolve()
    row = read_manifest_row(
        resolve(project_root, arguments.manifest), arguments.utterance_id
    )
    noisy = read_audio(resolve(project_root, str(row["noisy"])))
    if noisy.sample_rate != 16_000:
        raise ValueError("the frozen classical diagnostic expects 16 kHz input")

    enhancer = ClassicalEnhancer(
        wiener_config=WienerConfig(alpha_dd=0.92, gain_floor=0.20)
    )
    analysis = enhancer.stft.analyze(noisy.samples)
    result = enhancer.enhance(noisy.samples, method="mcra_dd_wiener")
    noisy_power_db = power_to_db(np.abs(analysis.spectrum) ** 2)
    noise_psd_db = power_to_db(result.diagnostics.noise_psd)
    gain_db = 20.0 * np.log10(
        np.maximum(result.diagnostics.gain, 1e-12)
    )

    frame_count, bin_count = noisy_power_db.shape
    if result.diagnostics.noise_psd.shape != (frame_count, bin_count):
        raise RuntimeError("MCRA PSD does not match the noisy STFT grid")
    time = np.arange(frame_count) * enhancer.stft.config.hop_length / noisy.sample_rate
    frequency = np.fft.rfftfreq(
        enhancer.stft.config.fft_length, 1.0 / noisy.sample_rate
    )
    shared_low = float(np.percentile(noisy_power_db, 2.0))
    shared_high = float(np.percentile(noisy_power_db, 99.5))

    figure, axes = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True, sharey=True, constrained_layout=True
    )
    labels = (
        "Noisy STFT power |Y(k, t)|²",
        "MCRA estimated noise PSD λₙ(k, t)",
        "Decision-directed Wiener magnitude gain",
    )
    images = []
    for axis, matrix, label in zip(
        axes[:2], (noisy_power_db, noise_psd_db), labels[:2], strict=True
    ):
        image = axis.pcolormesh(
            time,
            frequency,
            matrix.T,
            shading="auto",
            cmap="magma",
            vmin=shared_low,
            vmax=shared_high,
        )
        images.append(image)
        axis.set(title=label, ylabel="Frequency (Hz)", ylim=(0, 8_000))
    power_colorbar = figure.colorbar(images[0], ax=axes[:2], pad=0.01)
    power_colorbar.set_label("FFT-bin power / PSD (dB, same scale)")

    gain_image = axes[2].pcolormesh(
        time,
        frequency,
        gain_db.T,
        shading="auto",
        cmap="viridis",
        vmin=20.0 * np.log10(0.20),
        vmax=0.0,
    )
    axes[2].set(
        title=labels[2], xlabel="Time (s)", ylabel="Frequency (Hz)", ylim=(0, 8_000)
    )
    gain_colorbar = figure.colorbar(gain_image, ax=axes[2], pad=0.01)
    gain_colorbar.set_label("Magnitude gain (dB)")
    figure.suptitle(
        f"{row['id']} | noise={row.get('noise', 'unknown')}, "
        f"SNR={row.get('snr_db', 'unknown')} dB",
        fontsize=14,
    )

    output = resolve(project_root, arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
