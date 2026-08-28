from __future__ import annotations

import pytest

from robust_asr.models.paraformer_inference import _extract_paraformer_text


def test_extract_paraformer_text_accepts_one_result() -> None:
    result = [{"key": "u1", "text": " 测试语音 "}]

    assert _extract_paraformer_text(result) == "测试语音"


@pytest.mark.parametrize(
    "result",
    (
        {"text": "测试"},
        [],
        [{"text": "甲"}, {"text": "乙"}],
        [{"value": "测试"}],
        [{"text": 7}],
    ),
)
def test_extract_paraformer_text_rejects_ambiguous_results(result: object) -> None:
    with pytest.raises(RuntimeError):
        _extract_paraformer_text(result)
