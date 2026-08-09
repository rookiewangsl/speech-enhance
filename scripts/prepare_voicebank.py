"""Safely prepare 16 kHz VoiceBank + DEMAND paired evaluation protocols."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from speech_frontend.dataset import (
    PairedUtterance,
    discover_voicebank_pairs,
    prepare_pair,
    read_voicebank_condition_log,
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
    records: list[PairedUtterance],
    *,
    data_root: Path,
    limit: int | None,
    source_partition: str,
    workers: int,
    conditions: dict[str, dict[str, str | float]],
) -> None:
    if limit is not None:
        records = records[:limit]
    project_root = data_root.resolve().parent

    def prepare_record(record: PairedUtterance) -> dict:
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
        try:
            item.update(conditions[record.utterance_id])
        except KeyError as error:
            raise ValueError(
                f"missing official noise condition for {record.utterance_id}"
            ) from error
        return item

    manifest: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        prepared = executor.map(prepare_record, records)
        for index, item in enumerate(prepared, start=1):
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
        "--workers",
        type=int,
        default=4,
        help="Concurrent resampling workers (default: 4).",
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
    if arguments.workers <= 0:
        raise ValueError("workers must be positive")

    raw = arguments.data_root / "raw" / "voicebank_demand"
    log_archive = raw / "logfiles.zip"
    test_conditions = read_voicebank_condition_log(
        log_archive, "log_testset.txt"
    )
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
            workers=arguments.workers,
            conditions=test_conditions,
        )
        prepare_split(
            "holdout",
            holdout,
            data_root=arguments.data_root,
            limit=arguments.limit_per_split,
            source_partition="official_test",
            workers=arguments.workers,
            conditions=test_conditions,
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
    train_conditions = read_voicebank_condition_log(
        log_archive, "log_trainset_28spk.txt"
    )
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
    for name, records, source, conditions in (
        (
            "development",
            development,
            "official_train28",
            train_conditions,
        ),
        (
            "validation",
            validation,
            "official_train28",
            train_conditions,
        ),
        ("official_test", official_test, "official_test", test_conditions),
    ):
        prepare_split(
            name,
            records,
            data_root=arguments.data_root,
            limit=arguments.limit_per_split,
            source_partition=source,
            workers=arguments.workers,
            conditions=conditions,
        )
    print(
        f"Full protocol: train pairs={len(train_pairs)}, "
        f"development={len(development)}, validation={len(validation)}, "
        f"official_test={len(official_test)}."
    )


if __name__ == "__main__":
    main()
