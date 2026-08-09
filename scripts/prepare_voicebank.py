"""Safely prepare 16 kHz VoiceBank + DEMAND paired evaluation protocols."""

from __future__ import annotations

import argparse
from pathlib import Path

from speech_frontend.dataset import (
    discover_voicebank_pairs,
    prepare_pair,
    safe_extract_wav_zip,
    split_pairs_by_speaker,
    split_pairs_by_speaker_count,
    write_jsonl,
)


def extract_if_needed(archive: Path, destination: Path) -> None:
    if destination.exists():
        return
    safe_extract_wav_zip(archive, destination)


def prepare_split(
    split_name: str,
    records: list,
    *,
    data_root: Path,
    limit: int | None,
    source_partition: str,
) -> None:
    if limit is not None:
        records = records[:limit]
    project_root = data_root.resolve().parent
    manifest: list[dict] = []
    for index, record in enumerate(records, start=1):
        clean = (
            data_root
            / "processed"
            / "voicebank"
            / "clean"
            / f"{record.utterance_id}.wav"
        )
        noisy = (
            data_root
            / "processed"
            / "voicebank"
            / "noisy"
            / f"{record.utterance_id}.wav"
        )
        item = prepare_pair(record, clean, noisy)
        for key in ("clean", "noisy"):
            item[key] = Path(item[key]).resolve().relative_to(
                project_root
            ).as_posix()
        item["split"] = split_name
        item["source_partition"] = source_partition
        manifest.append(item)
        if index % 100 == 0 or index == len(records):
            print(f"{split_name}: prepared {index}/{len(records)}")
    write_jsonl(
        manifest,
        data_root / "manifests" / f"{split_name}.jsonl",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dev-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--protocol",
        choices=("legacy-test", "full28"),
        default="legacy-test",
        help=(
            "legacy-test reproduces the old 2-speaker internal split; full28 "
            "uses 20/8 training speakers plus the untouched official test set"
        ),
    )
    parser.add_argument(
        "--development-speakers",
        type=int,
        default=20,
        help="Number of 28-speaker training identities used for development.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Prepare only this many utterances per split for a smoke test.",
    )
    arguments = parser.parse_args()

    raw = arguments.data_root / "raw" / "voicebank_demand"
    extracted = raw / "extracted"
    if arguments.protocol == "legacy-test":
        clean_directory = extracted / "clean"
        noisy_directory = extracted / "noisy"
        extract_if_needed(raw / "clean_testset_wav.zip", clean_directory)
        extract_if_needed(raw / "noisy_testset_wav.zip", noisy_directory)
        pairs = discover_voicebank_pairs(clean_directory, noisy_directory)
        dev, holdout = split_pairs_by_speaker(
            pairs,
            dev_fraction=arguments.dev_fraction,
            seed=arguments.seed,
        )
        prepare_split(
            "dev",
            dev,
            data_root=arguments.data_root,
            limit=arguments.limit_per_split,
            source_partition="official_test",
        )
        prepare_split(
            "holdout",
            holdout,
            data_root=arguments.data_root,
            limit=arguments.limit_per_split,
            source_partition="official_test",
        )
        print(
            f"Found {len(pairs)} pairs; dev={len(dev)}, holdout={len(holdout)}. "
            "The split is speaker-disjoint but is not the official benchmark split."
        )
        return

    train_clean = extracted / "train28" / "clean"
    train_noisy = extracted / "train28" / "noisy"
    test_clean = extracted / "test" / "clean"
    test_noisy = extracted / "test" / "noisy"
    extract_if_needed(raw / "clean_trainset_28spk_wav.zip", train_clean)
    extract_if_needed(raw / "noisy_trainset_28spk_wav.zip", train_noisy)
    extract_if_needed(raw / "clean_testset_wav.zip", test_clean)
    extract_if_needed(raw / "noisy_testset_wav.zip", test_noisy)

    train_pairs = discover_voicebank_pairs(train_clean, train_noisy)
    official_test = discover_voicebank_pairs(test_clean, test_noisy)
    development, validation = split_pairs_by_speaker_count(
        train_pairs,
        first_speaker_count=arguments.development_speakers,
        seed=arguments.seed,
    )
    for name, records, source in (
        ("development", development, "official_train28"),
        ("validation", validation, "official_train28"),
        ("official_test", official_test, "official_test"),
    ):
        prepare_split(
            name,
            records,
            data_root=arguments.data_root,
            limit=arguments.limit_per_split,
            source_partition=source,
        )
    print(
        f"Full protocol: train pairs={len(train_pairs)}, "
        f"development={len(development)}, validation={len(validation)}, "
        f"official_test={len(official_test)}."
    )


if __name__ == "__main__":
    main()
