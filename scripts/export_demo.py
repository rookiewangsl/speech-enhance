"""Export the frozen noisy/classical/RNNoise listening set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_rnnoise import enhance_in_chunks
from speech_frontend.audio import AudioData, read_audio, write_audio
from speech_frontend.enhancement.wiener import WienerConfig
from speech_frontend.metrics import si_sdr, stoi
from speech_frontend.pipeline import ClassicalEnhancer
from speech_frontend.rnnoise import RNNoiseLibrary
from speech_frontend.rnnoise.controller import (
    CorrectionAwareConfig,
    correction_aware_mix,
    fixed_residual_mix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/listening_set.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/final/listening"),
    )
    parser.add_argument("--library", type=Path)
    arguments = parser.parse_args()

    protocol = json.loads(arguments.config.read_text(encoding="utf-8"))
    library = RNNoiseLibrary(arguments.library)
    classical = ClassicalEnhancer(
        wiener_config=WienerConfig(
            alpha_dd=0.92,
            gain_floor=0.10,
        )
    )
    rows: list[dict[str, str | float]] = []

    for item in protocol["items"]:
        file_id = item["id"]
        clean = read_audio(
            Path("data/processed/voicebank/clean") / f"{file_id}.wav"
        )
        noisy = read_audio(
            Path("data/processed/voicebank/noisy") / f"{file_id}.wav"
        )
        r1 = classical.enhance(
            noisy.samples,
            method="mcra_om_lsa",
        ).samples.astype(np.float32)
        r3, vad_probability, _ = enhance_in_chunks(
            noisy.samples,
            library,
            chunk_size=137,
        )
        conservative = fixed_residual_mix(noisy.samples, r3, 0.50)
        aggressive_result = correction_aware_mix(
            noisy.samples,
            r3,
            vad_probabilities=vad_probability,
            config=CorrectionAwareConfig(
                correction_threshold_db=-9.0,
                speech_protection=0.30,
            ),
        )
        methods = {
            "clean_reference": clean.samples,
            "noisy": noisy.samples,
            "r1_classical_omlsa": r1,
            "r3_official_rnnoise": r3,
            "r4_conservative": conservative,
            "r4_aggressive": aggressive_result.samples,
        }
        item_root = arguments.output_root / file_id
        input_si_sdr = si_sdr(clean.samples, noisy.samples)
        input_stoi = stoi(
            clean.samples,
            noisy.samples,
            sample_rate=clean.sample_rate,
        )
        for method, samples in methods.items():
            write_audio(
                item_root / f"{method}.wav",
                AudioData(
                    np.asarray(samples, dtype=np.float32),
                    noisy.sample_rate,
                ),
            )
            rows.append(
                {
                    "file_id": file_id,
                    "split": item["split"],
                    "reason": item["reason"],
                    "method": method,
                    "si_sdri_db": (
                        si_sdr(clean.samples, samples) - input_si_sdr
                    ),
                    "stoi_improvement": (
                        stoi(
                            clean.samples,
                            samples,
                            sample_rate=clean.sample_rate,
                        )
                        - input_stoi
                    ),
                    "peak": float(np.max(np.abs(samples), initial=0.0)),
                }
            )
        print(f"exported {file_id}", flush=True)

    metrics_path = arguments.output_root / "listening_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (arguments.output_root / "README.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
