"""Check accounting relationships rather than asserting hardware throughput."""

import pytest

from scripts.benchmarks.estimate import estimate


def test_self_play_cost_accounts_for_both_roles_and_feedback() -> None:
    result = estimate(steps=1000, seconds_per_step=10, gpu_hourly_usd=0.3)
    assert result["feedback_calls_per_step"] == 10
    assert result["feedback_calls"] == 10000
    assert result["estimated_feedback_usd"] == pytest.approx(1.125)
    assert result["wall_hours"] == pytest.approx(10000 / 3600 + 0.25)
    assert result["estimated_total_usd"] == pytest.approx(
        result["estimated_gpu_usd"] + 1.125
    )
    with pytest.raises(ValueError):
        estimate(steps=0, seconds_per_step=1, gpu_hourly_usd=0.3)
