from __future__ import annotations

import numpy as np
import pytest

from robust_asr.frontend_metrics import _paired_mean_interval
from robust_asr.signal_metrics import (
    match_signal_lengths,
    scale_invariant_sdr,
)


def test_signal_length_matching_only_pads_the_shorter_signal() -> None:
    reference, estimate = match_signal_lengths(
        np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0, 3.0])
    )

    assert reference.tolist() == [1.0, 2.0, 0.0]
    assert estimate.tolist() == [1.0, 2.0, 3.0]


def test_si_sdr_prefers_scaled_exact_target_over_noisy_estimate() -> None:
    time = np.arange(16_000, dtype=np.float64) / 16_000
    target = np.sin(2 * np.pi * 220 * time)
    exact = scale_invariant_sdr(target, 0.2 * target)
    noisy = scale_invariant_sdr(
        target,
        0.2 * target + 0.05 * np.sin(2 * np.pi * 997 * time),
    )

    assert exact > 100
    assert noisy < exact


def test_paired_metric_interval_requires_exact_pairing() -> None:
    with pytest.raises(ValueError, match="identical"):
        _paired_mean_interval(
            {"a": 1.0},
            {"b": 2.0},
            draws=100,
            seed=1,
        )


def test_paired_metric_interval_has_expected_direction() -> None:
    interval = _paired_mean_interval(
        {"a": 1.0, "b": 2.0},
        {"a": 2.0, "b": 3.0},
        draws=1_000,
        seed=1,
    )

    assert interval["lower"] == pytest.approx(1.0)
    assert interval["median"] == pytest.approx(1.0)
    assert interval["upper"] == pytest.approx(1.0)
