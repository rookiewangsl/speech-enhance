"""Build controlled VAD mixtures from a paired VoiceBank manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from speech_frontend.audio import AudioData, read_audio, validate_aligned_pair, write_audio
from speech_frontend.dataset import write_jsonl
from speech_frontend.vad.synthetic import VADMixtureConfig, create_vad_mixture


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/vad"))
    parser.add_argument("--num-items", type=int, default=20)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--snr-db", nargs="+", type=float, default=[-5.0, 0.0, 5.0, 10.0, 15.0])
    parser.add_argument("--seed", type=int, default=20260724)
    arguments = parser.parse_args()
    if arguments.num_items <= 0:
        raise ValueError("num-items must be positive")

    project_root = arguments.project_root.resolve()
    rows = [
        json.loads(line)
        for line in resolve(project_root, str(arguments.manifest)).read_text().splitlines()
    ]
    if not rows:
        raise ValueError("manifest has no rows")
    clean_pool = []
    residual_pool = []
    sample_rate = None
    for row in rows:
        clean = read_audio(resolve(project_root, row["clean"]))
        noisy = read_audio(resolve(project_root, row["noisy"]))
        validate_aligned_pair(clean, noisy)
        sample_rate = clean.sample_rate if sample_rate is None else sample_rate
        if clean.sample_rate != sample_rate:
            raise ValueError("all source files must use the same sample rate")
        clean_pool.append(clean.samples)
        residual_pool.append(noisy.samples - clean.samples)

    output_root = resolve(project_root, str(arguments.output_root))
    rng = np.random.default_rng(arguments.seed)
    config = VADMixtureConfig(
        sample_rate=sample_rate,
        duration_seconds=arguments.duration_seconds,
    )
    manifest: list[dict] = []
    for index in range(arguments.num_items):
        snr_db = arguments.snr_db[index % len(arguments.snr_db)]
        mixture = create_vad_mixture(
            clean_pool,
            residual_pool,
            snr_db=snr_db,
            config=config,
            rng=rng,
        )
        identifier = f"mix_{index:06d}"
        clean_path = output_root / "clean" / f"{identifier}.wav"
        noisy_path = output_root / "noisy" / f"{identifier}.wav"
        write_audio(clean_path, AudioData(mixture.clean.astype("float32"), sample_rate))
        write_audio(noisy_path, AudioData(mixture.noisy.astype("float32"), sample_rate))
        manifest.append(
            {
                "id": identifier,
                "clean": clean_path.relative_to(project_root).as_posix(),
                "noisy": noisy_path.relative_to(project_root).as_posix(),
                "sample_rate": sample_rate,
                "snr_db": snr_db,
                "speech_intervals": [list(interval) for interval in mixture.speech_intervals],
                "source": "VoiceBank+DEMAND residual noise; controlled insertion labels",
            }
        )
    manifest_path = output_root / "manifest.jsonl"
    write_jsonl(manifest, manifest_path)
    print(f"created {len(manifest)} mixtures at {manifest_path}")


if __name__ == "__main__":
    main()
