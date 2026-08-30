#!/usr/bin/env python3
"""Aggregate per-condition number-normalization audits into fixed scopes."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robust_asr.paths import require_data_root
from robust_asr.scoring import CharacterErrorCounts

_METRICS = (
    "formal",
    "deterministic_contextual",
    "deterministic_digit_by_digit",
    "number_equivalent_diagnostic",
)
_SCOPES: dict[str, tuple[str, tuple[float, ...] | None]] = {
    "clean": ("clean", None),
    "robust_raw": ("raw", (0.4, 0.6, 0.8, 1.0)),
    "robust_m_wpe": ("m_wpe_10", (0.4, 0.6, 0.8, 1.0)),
    "heavy_raw": ("raw", (0.8, 1.0)),
    "heavy_m_wpe": ("m_wpe_10", (0.8, 1.0)),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--audit",
        action="append",
        required=True,
        metavar="LABEL=FILE",
        help="Repeat for each audit JSON basename.",
    )
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


def _basename(value: str, option: str) -> str:
    if Path(value).name != value:
        raise ValueError(f"{option} must be a basename")
    return value


def _audit_spec(value: str) -> tuple[str, str]:
    label, separator, filename = value.partition("=")
    if not separator or not label or not filename:
        raise ValueError("--audit must use LABEL=FILE")
    return label, _basename(filename, "--audit file")


def _counts(value: Mapping[str, Any]) -> CharacterErrorCounts:
    return CharacterErrorCounts(
        substitutions=int(value["substitutions"]),
        deletions=int(value["deletions"]),
        insertions=int(value["insertions"]),
        reference_characters=int(value["reference_characters"]),
    )


def _aggregate_counts(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = CharacterErrorCounts(
        substitutions=sum(_counts(value).substitutions for value in values),
        deletions=sum(_counts(value).deletions for value in values),
        insertions=sum(_counts(value).insertions for value in values),
        reference_characters=sum(
            _counts(value).reference_characters for value in values
        ),
    )
    return total.as_dict()


def _scope_summary(
    conditions: Sequence[Mapping[str, Any]],
    *,
    frontend: str,
    rt60_values: tuple[float, ...] | None,
) -> dict[str, Any] | None:
    selected = [
        condition
        for condition in conditions
        if str(condition["frontend"]) == frontend
        and (
            condition.get("target_rt60_seconds") is None
            if rt60_values is None
            else any(
                abs(float(condition["target_rt60_seconds"]) - value) < 1e-8
                for value in rt60_values
            )
        )
    ]
    if not selected:
        return None
    metrics = {
        metric: _aggregate_counts([condition[metric] for condition in selected])
        for metric in _METRICS
    }
    formal_cer = float(metrics["formal"]["cer"])
    return {
        "utterances": sum(int(condition["utterances"]) for condition in selected),
        "hypotheses_with_ascii_digits": sum(
            int(condition["hypotheses_with_ascii_digits"])
            for condition in selected
        ),
        "metrics": metrics,
        "cer_percent": {
            metric: 100.0 * float(value["cer"])
            for metric, value in metrics.items()
        },
        "shift_from_formal_pp": {
            metric: 100.0 * (float(value["cer"]) - formal_cer)
            for metric, value in metrics.items()
            if metric != "formal"
        },
    }


def summarize_audits(audits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for label, audit in audits.items():
        if int(audit.get("schema_version", 0)) != 2:
            raise ValueError(f"{label} does not use number-audit schema version 2")
        conditions = audit["conditions"]
        models[label] = {
            "result_rows": int(audit["result_rows"]),
            "scopes": {
                scope: _scope_summary(
                    conditions,
                    frontend=definition[0],
                    rt60_values=definition[1],
                )
                for scope, definition in _SCOPES.items()
            },
        }
    return {
        "schema_version": 1,
        "purpose": "secondary_number_normalization_contrast",
        "primary_metric_unchanged": True,
        "models": models,
    }


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    audits: dict[str, Mapping[str, Any]] = {}
    for spec in args.audit:
        label, filename = _audit_spec(spec)
        if label in audits:
            raise ValueError(f"duplicate audit label: {label}")
        audits[label] = json.loads(
            (root / "outputs" / filename).read_text(encoding="utf-8")
        )
    summary = summarize_audits(audits)
    output_name = _basename(args.output_name, "--output-name")
    destination = root / "outputs" / output_name
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(f"wrote number-normalization summary for {len(audits)} models: {destination}")


if __name__ == "__main__":
    main()
