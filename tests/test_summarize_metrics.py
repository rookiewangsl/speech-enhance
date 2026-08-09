from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "summarize_metrics_by_condition.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_metrics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_groups_by_method_noise_and_snr() -> None:
    rows = [
        {
            "method": "r3",
            "noise": "bus",
            "snr_db": "2.5",
            "si_sdri_db": "2.0",
            "stoi_improvement": "0.1",
        },
        {
            "method": "r3",
            "noise": "cafe",
            "snr_db": "2.5",
            "si_sdri_db": "-1.0",
            "stoi_improvement": "-0.1",
        },
    ]

    result = MODULE.summarize(rows)

    assert result["overall"]["r3"]["mean_si_sdri_db"] == 0.5
    assert result["by_noise"]["r3"]["bus"]["files"] == 1
    assert result["by_snr_db"]["r3"]["2.5"]["files"] == 2
