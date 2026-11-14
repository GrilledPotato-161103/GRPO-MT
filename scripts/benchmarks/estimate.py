"""Estimate translation self-play runtime and GPU/API cost from explicit assumptions."""

import argparse
import json
from typing import Any


def estimate(
    *,
    steps: int,
    seconds_per_step: float,
    gpu_hourly_usd: float,
    proposal_batch_size: int = 1,
    proposal_group_size: int = 2,
    solver_group_size: int = 4,
    valid_fraction: float = 1.0,
    judge_input_tokens: int = 350,
    judge_output_tokens: int = 100,
    input_usd_per_million: float = 0.15,
    output_usd_per_million: float = 0.60,
    setup_minutes: float = 10.0,
    validation_minutes: float = 5.0,
) -> dict[str, Any]:
    """Compute a transparent scenario; prices and throughput are supplied, not inferred.

    seconds_per_step includes feedback wait and all optimizer iterations. Every
    well-formed unique proposal is validated; valid proposals incur G solve calls.
    Excludes transport retries, judge reasoning tokens, storage, tax and transfers.
    """
    if steps < 1 or seconds_per_step <= 0 or gpu_hourly_usd < 0:
        raise ValueError("Need positive steps/throughput and nonnegative GPU price")
    if (
        min(proposal_batch_size, proposal_group_size, solver_group_size) < 1
        or not 0 <= valid_fraction <= 1
    ):
        raise ValueError("Invalid rollout groups or valid-task fraction")
    if (
        min(
            judge_input_tokens,
            judge_output_tokens,
            input_usd_per_million,
            output_usd_per_million,
            setup_minutes,
            validation_minutes,
        )
        < 0
    ):
        raise ValueError(
            "Token, price, setup and validation assumptions must be nonnegative"
        )
    proposals = proposal_batch_size * proposal_group_size
    calls_per_step = proposals + proposals * valid_fraction * solver_group_size
    calls = calls_per_step * steps
    api_usd = (
        calls
        * (
            judge_input_tokens * input_usd_per_million
            + judge_output_tokens * output_usd_per_million
        )
        / 1e6
    )
    hours = (steps * seconds_per_step / 60 + setup_minutes + validation_minutes) / 60
    return {
        "assumptions_only": True,
        "steps": steps,
        "seconds_per_step": seconds_per_step,
        "gpu_hourly_usd": gpu_hourly_usd,
        "wall_hours": hours,
        "feedback_calls_per_step": calls_per_step,
        "feedback_calls": calls,
        "estimated_gpu_usd": hours * gpu_hourly_usd,
        "estimated_feedback_usd": api_usd,
        "estimated_total_usd": hours * gpu_hourly_usd + api_usd,
        "excludes": [
            "storage",
            "network transfers",
            "tax",
            "API retries",
            "reasoning tokens",
        ],
    }


def main() -> None:
    """Print a JSON scenario suitable for saving with a benchmark report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seconds-per-step", type=float, required=True)
    parser.add_argument("--gpu-hourly-usd", type=float, required=True)
    parser.add_argument("--setup-minutes", type=float, default=10)
    parser.add_argument("--validation-minutes", type=float, default=5)
    args = parser.parse_args()
    print(json.dumps(estimate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
