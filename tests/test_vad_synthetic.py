from __future__ import annotations

import numpy as np

from speech_frontend.vad.metrics import binary_metrics, labels_from_intervals
from speech_frontend.vad.synthetic import VADMixtureConfig, create_vad_mixture


def test_synthetic_mixture_has_exact_intervals_and_requested_snr() -> None:
    speech = [np.full(16_000, 0.2)]
    noise = [np.tile(np.array([-0.1, 0.1]), 8_000)]
    config = VADMixtureConfig(
        duration_seconds=4.0,
        minimum_segments=1,
        maximum_segments=1,
        minimum_segment_seconds=1.0,
        maximum_segment_seconds=1.0,
        maximum_gap_seconds=0.5,
    )

    mixture = create_vad_mixture(
        speech,
        noise,
        snr_db=0.0,
        config=config,
        rng=np.random.default_rng(4),
    )

    assert mixture.clean.shape == mixture.noisy.shape == (64_000,)
    start, stop = mixture.speech_intervals[0]
    assert np.all(mixture.clean[:start] == 0.0)
    assert np.all(mixture.clean[stop:] == 0.0)
    residual = mixture.noisy - mixture.clean
    snr = 10.0 * np.log10(
        np.mean(mixture.clean[start:stop] ** 2)
        / np.mean(residual[start:stop] ** 2)
    )
    assert abs(snr - mixture.snr_db) < 1e-8


def test_projected_frame_labels_and_metrics_are_correct() -> None:
    target = labels_from_intervals(
        8,
        frame_length=4,
        hop_length=2,
        intervals=((3, 7),),
    )
    prediction = np.array([True, True, True, True, False, False, False, False])

    metrics = binary_metrics(target, prediction)

    np.testing.assert_array_equal(
        target,
        [True, True, True, True, False, False, False, False],
    )
    assert metrics.f1 == 1.0


def test_synthetic_layout_handles_many_random_multisegment_cases() -> None:
    speech = [np.linspace(-0.1, 0.1, 40_000)]
    noise = [np.tile(np.array([-0.1, 0.1]), 20_000)]
    config = VADMixtureConfig(duration_seconds=10.0)
    rng = np.random.default_rng(2026)

    for _ in range(100):
        mixture = create_vad_mixture(
            speech,
            noise,
            snr_db=0.0,
            config=config,
            rng=rng,
        )
        assert all(
            0 <= start < stop <= mixture.clean.size
            for start, stop in mixture.speech_intervals
        )
