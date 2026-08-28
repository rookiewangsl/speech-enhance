from robust_asr.decode_compare import compare_greedy_with_beam


def _greedy(utterance: str, frontend: str, rt60, hypothesis: str) -> dict:
    return {
        "utterance_id": utterance,
        "frontend": frontend,
        "target_rt60_seconds": rt60,
        "reference": "测试语音",
        "hypothesis": hypothesis,
    }


def _beam(utterance: str, condition: str, rt60, hypothesis: str) -> dict:
    return {
        "utterance_id": utterance,
        "condition": condition,
        "reverb_rt60_seconds": rt60,
        "reference": "测试语音",
        "hypothesis": hypothesis,
    }


def test_compare_greedy_with_beam_matches_three_audit_conditions() -> None:
    greedy = [
        _greedy("u", "clean", None, "测试"),
        _greedy("u", "raw", 0.8, "测式"),
        _greedy("u", "m_wpe_10", 0.8, "测试语"),
    ]
    beam = [
        _beam("u", "clean_level", None, "测试语音"),
        _beam("u", "reverb_raw", 0.8, "测试语"),
        _beam("u", "reverb_m_wpe_10", 0.8, "测试语音"),
    ]

    summary = compare_greedy_with_beam(greedy, beam, draws=20, seed=7)

    assert {row["condition"] for row in summary["comparisons"]} == {
        "clean_level",
        "reverb_raw",
        "reverb_m_wpe_10",
    }
    assert all(
        row["beam"]["errors"] < row["greedy"]["errors"]
        for row in summary["comparisons"]
    )
