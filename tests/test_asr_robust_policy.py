from speech_frontend.asr.robust_policy import decide_final, normalized_word_distance


def _attempt(anomalous: bool, name: str) -> dict[str, object]:
    return {"name": name, "diagnostic": {"anomalous": anomalous}}


def test_first_pass_is_preserved_when_normal() -> None:
    first = _attempt(False, "first")
    result = decide_final(
        condition="rnnoise_r3",
        first_pass=first,
        retry_attempts=[_attempt(False, "retry")],
        fallback_conditions={"rnnoise_r3"},
        noisy_result=None,
        abstention_enabled=True,
    )
    assert result == {"status": "accepted", "source": "first_pass", "attempt": first}


def test_first_passing_retry_is_selected() -> None:
    passing = _attempt(False, "temperature_0.4")
    result = decide_final(
        condition="rnnoise_r3",
        first_pass=_attempt(True, "first"),
        retry_attempts=[_attempt(True, "temperature_0.2"), passing],
        fallback_conditions={"rnnoise_r3"},
        noisy_result=None,
        abstention_enabled=True,
    )
    assert result["source"] == "temperature_retry"
    assert result["attempt"] is passing


def test_word_distance_and_noisy_consistency_gate() -> None:
    assert normalized_word_distance("this is a club", "at last a letter arrived") == 0.8
    assert normalized_word_distance("please call stella", "please call stella") == 0.0
    retry = _attempt(False, "retry")
    retry["hypothesis_normalized"] = "this is a club"
    noisy_attempt = _attempt(False, "noisy")
    noisy_attempt["hypothesis_normalized"] = "at last a letter arrived"
    result = decide_final(
        condition="rnnoise_r3",
        first_pass=_attempt(True, "first"),
        retry_attempts=[retry],
        fallback_conditions={"rnnoise_r3"},
        noisy_result={"final_status": "accepted", "final_attempt": noisy_attempt},
        abstention_enabled=True,
        retry_noisy_consistency_threshold=0.4,
    )
    assert result["source"] == "noisy_fallback"
    assert result["decision_details"]["retry_rejected_by_noisy_consistency"] is True


def test_enhanced_condition_falls_back_to_accepted_noisy() -> None:
    noisy_attempt = _attempt(False, "noisy")
    result = decide_final(
        condition="rnnoise_r3",
        first_pass=_attempt(True, "first"),
        retry_attempts=[_attempt(True, "retry")],
        fallback_conditions={"rnnoise_r3"},
        noisy_result={"final_status": "accepted", "final_attempt": noisy_attempt},
        abstention_enabled=True,
    )
    assert result["source"] == "noisy_fallback"
    assert result["attempt"] is noisy_attempt


def test_noisy_cannot_fall_back_to_itself_and_abstains() -> None:
    result = decide_final(
        condition="noisy",
        first_pass=_attempt(True, "first"),
        retry_attempts=[],
        fallback_conditions={"rnnoise_r3"},
        noisy_result=None,
        abstention_enabled=True,
    )
    assert result == {"status": "abstained", "source": "abstained", "attempt": None}
