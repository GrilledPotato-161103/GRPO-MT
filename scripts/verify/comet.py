"""Verify real COMETKiwi inference and record cold/warm latency on synthetic EN/VI pairs."""

import argparse
import json
import time
from pathlib import Path

import torch

from src.models.components.comet_reward import CometScorer


def verify(
    output: Path,
    device: str = "cpu",
    cache_dir: str | None = None,
    model: str = "Unbabel/wmt22-cometkiwi-da",
) -> dict[str, object]:
    """Compare correct/unrelated candidates in both directions without private data."""
    torch.set_num_threads(4)
    scorer = CometScorer(
        model_name=model, device=device, batch_size=4, cache_dir=cache_dir
    )
    english, vietnamese = "This is a book.", "Đây là một cuốn sách."
    sources = [english, english, vietnamese, vietnamese]
    candidates = [vietnamese, "Trời đang mưa.", english, "It is raining."]
    started = time.perf_counter()
    scores = scorer(sources, candidates)
    cold_seconds = time.perf_counter() - started
    cold_phases = dict(scorer.last_timings)
    started = time.perf_counter()
    warm_scores = scorer(sources, candidates)
    warm_seconds = time.perf_counter() - started
    if not (scores[0] > scores[1] and scores[2] > scores[3]):
        raise AssertionError(
            "COMETKiwi did not rank the expected synthetic translations higher"
        )
    report: dict[str, object] = {
        "model": scorer.model_name,
        "device": device,
        "scores": scores,
        "warm_scores": warm_scores,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "warm_samples_per_second": 4 / warm_seconds,
        "cold_phases": cold_phases,
        "warm_phases": dict(scorer.last_timings),
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in scorer.model.parameters()
        ),
        "reference_free": True,
        "synthetic_quality_ordering_passed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Run a credential-aware local or rental smoke test without printing secrets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--model", default="Unbabel/wmt22-cometkiwi-da")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.output, args.device, args.cache_dir, args.model), indent=2
        )
    )


if __name__ == "__main__":
    main()
