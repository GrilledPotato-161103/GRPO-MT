"""Length buckets for raw translation pairs, without changing split membership."""

import random
from collections.abc import Iterator, Sequence
from typing import Any

from torch.utils.data import Sampler

from src.data.components.translation_prompt import translation_prompts
from src.data.components.multilingual_tokens import multilingual_ids


def padded_length_batches(
    lengths: Sequence[tuple[int, int]],
    batch_size: int,
    token_budget: int | None = None,
    multiplier: int = 1,
) -> list[list[int]]:
    """Pack in order using N * (max source + max target), not unpadded tokens."""
    if (
        batch_size < 1
        or multiplier < 1
        or (token_budget is not None and token_budget < 1)
    ):
        raise ValueError(
            "Batch size, candidate multiplier and token budget must be positive"
        )
    batches: list[list[int]] = []
    current: list[int] = []
    source_max = target_max = 0
    for index, (source, target) in enumerate(lengths):
        if source < 1 or target < 1:
            raise ValueError("Source and target lengths must be positive")
        if token_budget is not None and multiplier * (source + target) > token_budget:
            raise ValueError(
                "One example exceeds the padded token budget; increase the budget or lower the caps"
            )
        next_source, next_target = max(source_max, source), max(target_max, target)
        if current and (
            len(current) == batch_size
            or (
                token_budget is not None
                and multiplier * (len(current) + 1) * (next_source + next_target)
                > token_budget
            )
        ):
            batches.append(current)
            current, source_max, target_max = [], 0, 0
        current.append(index)
        source_max, target_max = max(source_max, source), max(target_max, target)
    if current:
        batches.append(current)
    return batches


def translation_lengths(
    rows: list[dict[str, str]],
    tokenizer: Any,
    source_cap: int,
    target_cap: int,
    prompt_template: str,
    source_lang: str = "English",
    target_lang: str = "Vietnamese",
    use_chat_template: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
    language_codes: dict[str, str] | None = None,
) -> list[tuple[int, int]]:
    """Estimate capped prompt/target lengths; no augmentation or RNG is consumed."""
    if language_codes:
        src = [row.get("source_lang", source_lang) for row in rows]
        tgt = [row.get("target_lang", target_lang) for row in rows]
        sources = multilingual_ids(tokenizer, [r["source"] for r in rows], src, tgt, language_codes,
                                   truncation=True, max_length=source_cap)
        targets = multilingual_ids(tokenizer, [r["reference"] for r in rows], src, tgt, language_codes,
                                   targets=True, truncation=True, max_length=target_cap)
        return [(len(a), len(b)) for a, b in zip(sources, targets, strict=True)]
    prompts = translation_prompts(
        tokenizer,
        [row["source"] for row in rows],
        [row.get("source_lang", source_lang) for row in rows],
        [row.get("target_lang", target_lang) for row in rows],
        prompt_template,
        use_chat_template,
        chat_template_kwargs,
    )
    options = {"add_special_tokens": False} if use_chat_template else {}
    sources = tokenizer(prompts, truncation=True, max_length=source_cap, **options)[
        "input_ids"
    ]
    targets = tokenizer(
        [
            row["reference"] if use_chat_template else " " + row["reference"]
            for row in rows
        ],
        add_special_tokens=False,
        truncation=True,
        max_length=max(1, target_cap - 1),
    )["input_ids"]
    return [
        (max(1, len(source)), min(target_cap, len(target) + 1))
        for source, target in zip(sources, targets, strict=True)
    ]


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Bucket by length first, then shuffle completed batches every epoch.

    Randomizing ties within narrow length bins changes batch membership without
    mixing arbitrary short and long examples. Every index occurs exactly once.
    """

    def __init__(
        self,
        lengths: Sequence[tuple[int, int]],
        batch_size: int,
        bucket_width: int = 16,
        seed: int = 42,
        token_budget: int | None = None,
        candidate_multiplier: int = 1,
        reserve_target_tokens: int = 0,
        reserve_source_tokens: int = 0,
    ) -> None:
        if (
            not lengths
            or bucket_width < 1
            or min(reserve_target_tokens, reserve_source_tokens) < 0
        ):
            raise ValueError(
                "Length buckets require examples, positive width and nonnegative target reserve"
            )
        self.lengths = list(lengths)
        self.batch_size, self.bucket_width, self.seed = batch_size, bucket_width, seed
        self.token_budget, self.candidate_multiplier = (
            token_budget,
            candidate_multiplier,
        )
        self.reserve_target_tokens, self.epoch = reserve_target_tokens, 0
        self.reserve_source_tokens = reserve_source_tokens
        self._cached_epoch = -1
        self._cached_batches: list[list[int]] = []
        # Lightning discovers set_epoch through batch_sampler.sampler.
        self.sampler = self
        self._batches()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        if self._cached_epoch == self.epoch:
            return self._cached_batches
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.lengths)))
        rng.shuffle(order)
        order.sort(
            key=lambda i: (
                self.lengths[i][0] // self.bucket_width,
                self.lengths[i][1] // self.bucket_width,
            )
        )
        lengths = [
            (
                max(self.lengths[i][0], self.reserve_source_tokens),
                max(self.lengths[i][1], self.reserve_target_tokens),
            )
            for i in order
        ]
        batches = [
            [order[i] for i in batch]
            for batch in padded_length_batches(
                lengths, self.batch_size, self.token_budget, self.candidate_multiplier
            )
        ]
        rng.shuffle(batches)
        self._cached_epoch, self._cached_batches = self.epoch, batches
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())


class TranslationCollator:
    """Sort actual augmented rows together, keeping every metadata column aligned."""

    def __init__(self, tokenizer: Any, **length_options: Any) -> None:
        self.tokenizer, self.length_options = tokenizer, length_options

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, list[str]]:
        if not rows or any(row.keys() != rows[0].keys() for row in rows):
            raise ValueError("Translation batches require rows with matching fields")
        lengths = translation_lengths(rows, self.tokenizer, **self.length_options)
        order = sorted(range(len(rows)), key=lambda i: (sum(lengths[i]), lengths[i][0]))
        return {key: [rows[i][key] for i in order] for key in rows[0]}


class PairedBatchSampler(Sampler[list[int]]):
    """Keep both directions atomic, including shuffled or length-bucketed batches."""

    def __init__(
        self,
        pairs: int,
        batch_size: int,
        seed: int = 42,
        shuffle: bool = True,
        pair_sampler: LengthBucketBatchSampler | None = None,
    ) -> None:
        if pairs < 1 or batch_size < 2 or batch_size % 2:
            raise ValueError(
                "Paired batching requires pairs and an even batch size >=2"
            )
        self.pairs, self.batch_size, self.seed = pairs, batch_size, seed
        self.shuffle, self.pair_sampler, self.epoch = shuffle, pair_sampler, 0
        self.sampler = self

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if self.pair_sampler is not None:
            self.pair_sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return (
            len(self.pair_sampler)
            if self.pair_sampler is not None
            else (self.pairs + self.batch_size // 2 - 1) // (self.batch_size // 2)
        )

    def __iter__(self) -> Iterator[list[int]]:
        if self.pair_sampler is not None:
            batches = list(self.pair_sampler)
        else:
            order = list(range(self.pairs))
            if self.shuffle:
                random.Random(self.seed + self.epoch).shuffle(order)
            size = self.batch_size // 2
            batches = [
                order[start : start + size] for start in range(0, len(order), size)
            ]
        for batch in batches:
            yield [view for pair in batch for view in (2 * pair, 2 * pair + 1)]
