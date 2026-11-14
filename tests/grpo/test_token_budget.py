"""Variable batch packing must respect padded memory and preserve every example."""

import pytest

from src.data.components.token_budget import padded_token_batches


@pytest.mark.parametrize(
    "lengths,budget",
    [([10, 100, 10, 20], 120), ([1] * 100, 16), ([640, 40, 60, 80, 640], 1280)],
)
def test_token_budget_preserves_order_and_bounds(
    lengths: list[int], budget: int
) -> None:
    batches = padded_token_batches(lengths, budget)
    assert [index for batch in batches for index in batch] == list(range(len(lengths)))
    assert all(
        len(batch) * max(lengths[index] for index in batch) <= budget
        for batch in batches
    )
    assert all(
        (len(batch) + 1)
        * max(max(lengths[index] for index in batch), lengths[batches[position + 1][0]])
        > budget
        for position, batch in enumerate(batches[:-1])
    )


def test_token_budget_rejects_oversized_example() -> None:
    with pytest.raises(ValueError):
        padded_token_batches([641], 640)
