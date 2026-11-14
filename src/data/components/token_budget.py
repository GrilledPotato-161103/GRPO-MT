"""Pack actual augmented SFT examples under a padded-token budget."""

from typing import Any

from torch.utils.data import Dataset


class PreparedTranslationDataset(Dataset[dict[str, str]]):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def padded_token_batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Keep original order, with batch size times maximum length within budget."""
    if budget < 1 or any(length < 1 or length > budget for length in lengths):
        raise ValueError("Every example must fit the positive token budget")
    batches: list[list[int]] = []
    current: list[int] = []
    maximum = 0
    for index, length in enumerate(lengths):
        candidate_maximum = max(maximum, length)
        if current and (len(current) + 1) * candidate_maximum > budget:
            batches.append(current)
            current, maximum = [], 0
        current.append(index)
        maximum = max(maximum, length)
    if current:
        batches.append(current)
    return batches


def prepare_token_batches(
    rows: list[dict[str, str]],
    tokenizer: Any,
    budget: int,
    source_cap: int,
    target_cap: int,
    prompt_template: str,
) -> tuple[PreparedTranslationDataset, list[list[int]]]:
    prompts = [
        prompt_template.format(
            source=row["source"],
            source_lang=row.get("source_lang", "English"),
            target_lang=row.get("target_lang", "Vietnamese"),
        )
        for row in rows
    ]
    sources = tokenizer(
        prompts, add_special_tokens=False, truncation=True, max_length=source_cap
    )["input_ids"]
    targets = tokenizer(
        [row["reference"] for row in rows],
        add_special_tokens=False,
        truncation=True,
        max_length=target_cap - 1,
    )["input_ids"]
    lengths = [
        len(source) + len(target) + 1
        for source, target in zip(sources, targets, strict=True)
    ]
    return PreparedTranslationDataset(rows), padded_token_batches(lengths, budget)
