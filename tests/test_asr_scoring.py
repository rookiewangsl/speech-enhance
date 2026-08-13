from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from speech_frontend.asr.scoring import aggregate_error_counts, score_text


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_asr_metrics.py"
SPEC = importlib.util.spec_from_file_location("summarize_asr_metrics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_word_alignment_reports_substitution_deletion_and_insertion() -> None:
    score = score_text("a b c d e f", "a x c y d e")

    assert score.substitutions == 1
    assert score.deletions == 1
    assert score.insertions == 1
    assert score.reference_words == 6
    assert score.wer == 0.5


def test_empty_hypothesis_is_all_deletions() -> None:
    score = score_text("please call stella", "")

    assert score.as_dict() == {
        "substitutions": 0,
        "deletions": 3,
        "insertions": 0,
        "reference_words": 3,
        "errors": 3,
        "wer": 1.0,
        "substitution_rate": 0.0,
        "deletion_rate": 1.0,
        "insertion_rate": 0.0,
    }


def test_empty_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty reference"):
        score_text("", "hallucinated words")


def test_corpus_wer_is_not_mean_utterance_wer() -> None:
    long_score = score_text("one two three four", "one two three")
    short_score = score_text("five", "")

    corpus = aggregate_error_counts([long_score, short_score])

    assert corpus.wer == 2 / 5
    assert corpus.wer != (long_score.wer + short_score.wer) / 2


def _rows() -> list[dict[str, object]]:
    references = {
        "u1": ("one two three four", "bus", 0.0, "p1"),
        "u2": ("five six", "babble", 15.0, "p2"),
    }
    hypotheses = {
        "noisy": {"u1": "one two", "u2": "five"},
        "clean": {"u1": "one two three four", "u2": "five six"},
        "rnnoise_r3": {"u1": "one two three", "u2": ""},
    }
    rows: list[dict[str, object]] = []
    for condition, condition_hypotheses in hypotheses.items():
        for utterance_id, hypothesis in condition_hypotheses.items():
            reference, noise, snr_db, speaker_id = references[utterance_id]
            rows.append(
                {
                    "id": utterance_id,
                    "condition": condition,
                    "reference_normalized": reference,
                    "reference_raw": reference,
                    "reference_raw_sha256": hashlib.sha256(
                        reference.encode("utf-8")
                    ).hexdigest(),
                    "hypothesis_normalized": hypothesis,
                    "noise": noise,
                    "snr_db": snr_db,
                    "speaker_id": speaker_id,
                    "duration_seconds": 2.0,
                    "asr_seconds": 1.0,
                    "end_to_end_seconds": 1.25,
                }
            )
    return rows


def test_summary_groups_and_compares_against_noisy() -> None:
    scored = MODULE.score_rows(_rows())
    summary = MODULE.summarize(scored)

    assert summary["overall"]["noisy"]["wer"] == 3 / 6
    assert summary["overall"]["clean"]["wer"] == 0.0
    assert summary["overall"]["clean"]["absolute_wer_change_vs_noisy"] == -0.5
    assert summary["overall"]["clean"]["relative_wer_reduction_vs_noisy"] == 1.0
    assert summary["overall"]["rnnoise_r3"]["wer"] == 3 / 6
    assert summary["overall"]["rnnoise_r3"]["relative_wer_reduction_vs_noisy"] == 0.0
    assert summary["by_noise"]["babble"]["noisy"]["deletions"] == 1
    assert summary["by_snr_db"]["15.0"]["rnnoise_r3"]["wer"] == 1.0
    assert summary["by_speaker_id"]["p1"]["clean"]["wer"] == 0.0
    assert summary["by_reference_length_bin"]["1-4"]["clean"]["wer"] == 0.0
    assert summary["overall"]["noisy"]["asr_rtf"] == 0.5
    assert summary["overall"]["noisy"]["end_to_end_rtf"] == 0.625
    assert summary["bootstrap"]["draws"] == MODULE.DEFAULT_BOOTSTRAP_DRAWS
    assert summary["overall"]["noisy"][
        "paired_bootstrap_absolute_wer_change_vs_noisy_ci95"
    ] == {"lower": 0.0, "upper": 0.0}


def test_join_rejects_unpaired_condition() -> None:
    references = [
        {"id": "u1", "reference_normalized": "one"},
        {"id": "u2", "reference_normalized": "two"},
    ]
    hypotheses = [
        {"id": "u1", "condition": "noisy", "hypothesis_normalized": "one"},
        {"id": "u2", "condition": "noisy", "hypothesis_normalized": "two"},
        {"id": "u1", "condition": "clean", "hypothesis_normalized": "one"},
    ]

    with pytest.raises(ValueError, match="not paired"):
        MODULE.merge_references_and_hypotheses(references, hypotheses)


def test_scoring_rejects_asymmetric_normalization() -> None:
    with pytest.raises(ValueError, match="normalized text for both"):
        MODULE.score_rows(
            [
                {
                    "id": "u1",
                    "condition": "noisy",
                    "reference_raw": "Hello, world!",
                    "hypothesis_normalized": "hello world",
                }
            ]
        )


def test_scoring_rejects_mixed_raw_and_normalized_rows() -> None:
    with pytest.raises(ValueError, match="cannot mix"):
        MODULE.score_rows(
            [
                {
                    "id": "u1",
                    "condition": "noisy",
                    "reference_raw": "one",
                    "hypothesis_raw": "one",
                },
                {
                    "id": "u2",
                    "condition": "noisy",
                    "reference_normalized": "two",
                    "hypothesis_normalized": "two",
                },
            ]
        )


def test_cli_style_separate_jsonl_merge(tmp_path: Path) -> None:
    references_path = tmp_path / "references.jsonl"
    hypotheses_path = tmp_path / "hypotheses.jsonl"
    references_path.write_text(
        json.dumps({"id": "u1", "reference_normalized": "hello world"}) + "\n",
        encoding="utf-8",
    )
    hypotheses_path.write_text(
        json.dumps(
            {
                "id": "u1",
                "condition": "noisy",
                "hypothesis_normalized": "hello",
                "status": "completed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    merged = MODULE.merge_references_and_hypotheses(
        MODULE.read_jsonl(references_path), MODULE.read_jsonl(hypotheses_path)
    )

    assert merged[0]["reference_normalized"] == "hello world"
    assert MODULE.score_rows(merged)[0]["deletions"] == 1


def test_reference_manifest_may_be_a_superset_of_evaluation_ids() -> None:
    references = [
        {"id": "u1", "reference_normalized": "one"},
        {"id": "u2", "reference_normalized": "two"},
    ]
    hypotheses = [
        {"id": "u1", "condition": "noisy", "hypothesis_normalized": "one"},
        {"id": "u1", "condition": "clean", "hypothesis_normalized": "one"},
    ]

    merged = MODULE.merge_references_and_hypotheses(references, hypotheses)

    assert len(merged) == 2


def test_join_rejects_hypothesis_that_overrides_authoritative_reference() -> None:
    references = [{"id": "u1", "reference_raw": "official words"}]
    hypotheses = [
        {
            "id": "u1",
            "condition": "noisy",
            "reference_raw": "ASR-side replacement",
            "hypothesis_raw": "official words",
        }
    ]

    with pytest.raises(ValueError, match="authoritative reference_raw"):
        MODULE.merge_references_and_hypotheses(references, hypotheses)


def test_frozen_experiment_requires_all_conditions_and_one_identity() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="requires conditions"):
        MODULE.validate_frozen_experiment(rows)

    template_identity = {
        "model_sha256": "f" * 64,
        "asr_config_digest": "e" * 64,
        "evaluator_code_sha256": "d" * 64,
        "runtime_identity_digest": "c" * 64,
        "device": "cpu",
    }
    complete = []
    for row in rows:
        complete.append({**row, **template_identity})
    for row in [value for value in rows if value["condition"] == "clean"]:
        complete.append({**row, **template_identity, "condition": "mcra_dd_wiener"})

    MODULE.validate_frozen_experiment(complete)
    complete[0]["device"] = "mps"
    with pytest.raises(ValueError, match="disagree in device"):
        MODULE.validate_frozen_experiment(complete)

    for row in complete:
        row["device"] = "mps"
    with pytest.raises(ValueError, match="device=cpu"):
        MODULE.validate_frozen_experiment(complete)


def test_paired_bootstrap_is_deterministic_and_uses_utterance_pairs() -> None:
    scored = MODULE.score_rows(_rows())

    first = MODULE.summarize(scored, bootstrap_draws=101, bootstrap_seed=17)
    second = MODULE.summarize(
        list(reversed(scored)), bootstrap_draws=101, bootstrap_seed=17
    )

    first_ci = first["overall"]["clean"][
        "paired_bootstrap_absolute_wer_change_vs_noisy_ci95"
    ]
    second_ci = second["overall"]["clean"][
        "paired_bootstrap_absolute_wer_change_vs_noisy_ci95"
    ]
    assert first_ci == second_ci
    assert first_ci == {"lower": -0.5, "upper": -0.5}


def test_paired_bootstrap_rejects_mismatched_ids() -> None:
    scored = MODULE.score_rows(_rows())
    incomplete = [
        row
        for row in scored
        if not (row["condition"] == "clean" and row["id"] == "u2")
    ]

    with pytest.raises(ValueError, match="same paired utterance ids"):
        MODULE.summarize(incomplete, bootstrap_draws=10)


def test_summary_rejects_partial_or_nonfinite_timing() -> None:
    partial = MODULE.score_rows(_rows())
    partial[0].pop("asr_seconds")
    with pytest.raises(ValueError, match="partial timing"):
        MODULE.summarize(partial)

    nonfinite = MODULE.score_rows(_rows())
    nonfinite[0]["asr_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite and non-negative"):
        MODULE.summarize(nonfinite)


def test_summary_reports_utterance_rtf_tail_and_decode_anomalies() -> None:
    rows = _rows()
    for row in rows:
        if row["condition"] == "noisy" and row["id"] == "u1":
            row["asr_seconds"] = 0.2
            row["segments"] = [{"compression_ratio": 2.4}]
        elif row["condition"] == "noisy" and row["id"] == "u2":
            row["asr_seconds"] = 4.0
            row["segments"] = [
                {"compression_ratio": 1.2},
                {"compression_ratio": 2.4001},
            ]

    summary = MODULE.summarize(MODULE.score_rows(rows))
    noisy = summary["overall"]["noisy"]

    assert noisy["asr_rtf"] == pytest.approx(1.05)
    assert noisy["utterance_asr_rtf_median"] == pytest.approx(1.05)
    assert noisy["utterance_asr_rtf_p95"] == pytest.approx(1.905)
    assert noisy["utterance_asr_rtf_max"] == pytest.approx(2.0)
    assert noisy["compression_ratio_anomaly_utterances"] == 1
    assert noisy["compression_ratio_anomaly_segments"] == 1
    assert noisy["compression_ratio_anomaly_metadata_utterances"] == 2
    assert noisy["compression_ratio_anomaly_fraction"] == 0.5
    assert summary["scoring"]["decode_anomaly"] == {
        "rule": "any_segment_compression_ratio_greater_than_threshold",
        "compression_ratio_threshold": 2.4,
        "threshold_source": "cli_fallback",
        "cli_fallback": 2.4,
        "anomalies_are_retained_in_wer": True,
    }
    anomalous = next(
        row
        for row in MODULE.score_rows(rows)
        if row["condition"] == "noisy" and row["id"] == "u2"
    )
    assert anomalous["compression_ratio_anomaly"] is True
    assert anomalous["max_segment_compression_ratio"] == 2.4001


def test_row_config_threshold_provenance_wins_over_cli_fallback() -> None:
    rows = _rows()
    for row in rows:
        row["thresholds"] = {"compression_ratio_threshold": 3.0}
        row["segments"] = [{"compression_ratio": 2.5}]

    scored = MODULE.score_rows(rows, compression_ratio_threshold=2.4)
    summary = MODULE.summarize(scored, compression_ratio_threshold=2.4)

    assert not any(row["compression_ratio_anomaly"] for row in scored)
    assert summary["scoring"]["decode_anomaly"]["compression_ratio_threshold"] == 3.0
    assert summary["scoring"]["decode_anomaly"]["threshold_source"] == (
        "row_config_provenance"
    )


def test_partial_row_config_threshold_provenance_is_rejected() -> None:
    rows = _rows()
    rows[0]["decoding"] = {"compression_ratio_threshold": 2.4}

    with pytest.raises(ValueError, match="provenance is partial"):
        MODULE.score_rows(rows)


def test_atomic_jsonl_failure_preserves_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "utterances.jsonl"
    output.write_text("previous\n", encoding="utf-8")

    with pytest.raises(TypeError):
        MODULE.write_jsonl(output, [{"id": "u1"}, {"invalid": {1, 2}}])

    assert output.read_text(encoding="utf-8") == "previous\n"
    assert not output.with_name(output.name + ".part").exists()
