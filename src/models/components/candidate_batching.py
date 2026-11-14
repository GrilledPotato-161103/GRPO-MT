"""Plan padded candidate graphs while preserving rollout indices and context."""

from dataclasses import dataclass

import torch
from torch import Tensor

from src.data.components.length_batching import padded_length_batches


@dataclass
class CandidateBatch:
    indices: Tensor
    inputs: dict[str, Tensor]
    sequences: Tensor
    mask: Tensor
    valid_tokens: int
    scored_tokens: int

    @property
    def padded_tokens(self) -> int:
        return self.mask.shape[0] * (
            self.inputs["input_ids"].shape[1] + self.mask.shape[1]
        )


def candidate_batches(
    inputs: dict[str, Tensor],
    sequences: Tensor,
    mask: Tensor,
    context_mask: Tensor,
    is_encoder_decoder: bool,
    batch_size: int | None,
    token_budget: int | None,
    sort_by_length: bool,
    trim_padding: bool,
) -> list[CandidateBatch]:
    """Reorder rows only; full-source advantages are computed before this function."""
    count, width = mask.shape
    if count == 0 or width == 0:
        raise ValueError(
            "Candidate scoring requires nonempty sequences and completions"
        )
    source_mask = inputs["attention_mask"].bool()
    columns = torch.arange(source_mask.shape[1], device=mask.device)
    source_starts = torch.where(source_mask, columns, source_mask.shape[1]).amin(-1)
    source_ends = torch.where(source_mask, columns + 1, 0).amax(-1).clamp_min(1)
    target_columns = torch.arange(width, device=mask.device) + 1
    target_ends = (
        torch.where(context_mask | mask, target_columns, 0).amax(-1).clamp_min(1)
    )
    if not trim_padding:
        source_starts = torch.zeros_like(source_starts)
        source_ends = torch.full_like(source_ends, source_mask.shape[1])
        target_ends = torch.full_like(target_ends, width)
    # Copy shape metadata once; avoid a CUDA synchronization for every chunk.
    spans = (
        torch.stack(
            [
                source_starts,
                source_ends,
                target_ends,
                source_mask.sum(-1),
                (context_mask | mask).sum(-1),
                mask.sum(-1),
            ],
            dim=-1,
        )
        .cpu()
        .tolist()
    )
    lengths = [
        (max(1, (end if is_encoder_decoder else source_mask.shape[1]) - start), target)
        for start, end, target, *_ in spans
    ]
    order = list(range(count))
    if sort_by_length:
        order.sort(key=lambda i: (lengths[i][0] // 16, lengths[i][1], lengths[i][0]))
    # Greedy budgeting uses padded dimensions, not the sum of valid tokens.
    planned = padded_length_batches(
        [lengths[i] for i in order], batch_size or count, token_budget
    )
    result = []
    prefix = 1 if is_encoder_decoder else inputs["input_ids"].shape[1]
    for positions in planned:
        members = [order[i] for i in positions]
        indices = torch.tensor(members, device=mask.device)
        start = min(spans[i][0] for i in members)
        source_end = max(spans[i][1] for i in members) if is_encoder_decoder else prefix
        end = max(spans[i][2] for i in members)
        selected_inputs = {
            key: value[indices, start:source_end] for key, value in inputs.items()
        }
        selected_sequences = (
            sequences[indices, : 1 + end]
            if is_encoder_decoder
            else sequences[indices, start : prefix + end]
        )
        result.append(
            CandidateBatch(
                indices,
                selected_inputs,
                selected_sequences,
                mask[indices, :end],
                sum(spans[i][3] + spans[i][4] for i in members),
                sum(spans[i][5] for i in members),
            )
        )
    return result
