"""Audit whether paired ASR conclusions survive speaker and outlier removal."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ENHANCED_CONDITIONS = ("mcra_dd_wiener", "rnnoise_r3")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty object JSONL: {path}")
    return rows


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def _paired_delta(rows: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    included = [row for row in rows if row.get(score_field) is not None]
    if not included:
        raise ValueError(f"no rows contain {score_field}")
    reference_words = sum(int(row["reference_words"]) for row in included)
    condition_errors = sum(int(row[score_field]["errors"]) for row in included)
    noisy_errors = sum(int(row["paired_v1_noisy"]["errors"]) for row in included)
    delta = (condition_errors - noisy_errors) / reference_words
    return {
        "utterances": len(rows),
        "included_utterances": len(included),
        "coverage": len(included) / len(rows),
        "reference_words": reference_words,
        "condition_errors": condition_errors,
        "paired_noisy_errors": noisy_errors,
        "condition_wer": condition_errors / reference_words,
        "paired_noisy_wer": noisy_errors / reference_words,
        "absolute_wer_change_vs_paired_noisy": delta,
    }


def analyze_field(rows: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    base = _paired_delta(rows, score_field)
    included = [row for row in rows if row.get(score_field) is not None]
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in included:
        by_speaker[str(row["speaker_id"])].append(row)
    speakers = {speaker: _paired_delta(group, score_field) for speaker, group in sorted(by_speaker.items())}
    overall_sign = _sign(float(base["absolute_wer_change_vs_paired_noisy"]))
    direction_counts = {
        "better": sum(item["absolute_wer_change_vs_paired_noisy"] < 0 for item in speakers.values()),
        "equal": sum(item["absolute_wer_change_vs_paired_noisy"] == 0 for item in speakers.values()),
        "worse": sum(item["absolute_wer_change_vs_paired_noisy"] > 0 for item in speakers.values()),
    }

    leave_one_out: dict[str, dict[str, Any]] = {}
    for speaker in speakers:
        kept = [row for row in included if str(row["speaker_id"]) != speaker]
        leave_one_out[speaker] = _paired_delta(kept, score_field)

    most_influential = max(
        included,
        key=lambda row: abs(int(row[score_field]["errors"]) - int(row["paired_v1_noisy"]["errors"])),
    )
    without_influential = _paired_delta(
        [row for row in included if row is not most_influential], score_field
    )
    babble = _paired_delta([row for row in included if row.get("noise") == "babble"], score_field)
    non_babble = _paired_delta([row for row in included if row.get("noise") != "babble"], score_field)

    return {
        **base,
        "speaker_results": speakers,
        "speaker_direction_counts": direction_counts,
        "leave_one_speaker_out": leave_one_out,
        "all_leave_one_speaker_out_same_direction": all(
            _sign(float(item["absolute_wer_change_vs_paired_noisy"])) == overall_sign
            for item in leave_one_out.values()
        ),
        "most_influential_utterance": {
            "id": most_influential["id"],
            "speaker_id": most_influential["speaker_id"],
            "noise": most_influential["noise"],
            "snr_db": most_influential["snr_db"],
            "condition_errors": most_influential[score_field]["errors"],
            "paired_noisy_errors": most_influential["paired_v1_noisy"]["errors"],
        },
        "without_most_influential": without_influential,
        "without_most_influential_same_direction": _sign(
            float(without_influential["absolute_wer_change_vs_paired_noisy"])
        ) == overall_sign,
        "babble": babble,
        "non_babble": non_babble,
        "babble_and_non_babble_same_direction": all(
            _sign(float(item["absolute_wer_change_vs_paired_noisy"])) == overall_sign
            for item in (babble, non_babble)
        ),
    }


def analyze(rows: list[dict[str, Any]], v1_summary: dict[str, Any]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition"))].append(row)
    if any(len(by_condition[condition]) == 0 for condition in ENHANCED_CONDITIONS):
        raise ValueError("stability analysis requires both enhanced conditions")
    output: dict[str, Any] = {"schema_version": 1, "conditions": {}}
    for condition in ENHANCED_CONDITIONS:
        condition_rows = sorted(by_condition[condition], key=lambda row: str(row["id"]))
        first_pass = analyze_field(condition_rows, "first_pass")
        first_pass["paired_bootstrap_ci95"] = v1_summary["overall"][condition].get(
            "paired_bootstrap_absolute_wer_change_vs_noisy_ci95"
        )
        output["conditions"][condition] = {
            "v1_first_pass": first_pass,
            "v2_final": analyze_field(condition_rows, "final"),
        }
    return output


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utterances", type=Path, required=True)
    parser.add_argument("--v1-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(read_jsonl(args.utterances), read_json(args.v1_summary))
    _atomic_write(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
