from __future__ import annotations

import numpy as np
import pytest

from speech_frontend.audio import AudioData, validate_aligned_pair


def test_validate_aligned_pair_accepts_matching_audio() -> None:
    clean = AudioData(np.zeros(160, dtype=np.float32), 16_000)
    noisy = AudioData(np.ones(160, dtype=np.float32), 16_000)

    validate_aligned_pair(clean, noisy)


def test_validate_aligned_pair_rejects_mismatched_length() -> None:
    clean = AudioData(np.zeros(160, dtype=np.float32), 16_000)
    noisy = AudioData(np.zeros(159, dtype=np.float32), 16_000)

    with pytest.raises(ValueError, match="lengths"):
        validate_aligned_pair(clean, noisy)


def test_validate_aligned_pair_rejects_non_finite_audio() -> None:
    clean = AudioData(np.array([0.0, np.nan], dtype=np.float32), 16_000)
    noisy = AudioData(np.zeros(2, dtype=np.float32), 16_000)

    with pytest.raises(ValueError, match="NaN"):
        validate_aligned_pair(clean, noisy)
