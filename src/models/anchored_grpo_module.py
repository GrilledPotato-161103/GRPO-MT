"""GRPO for unrejected groups; verified correction contrasts for failed groups."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from src.data.components.multilingual_tokens import multilingual_ids
from src.models.components.correction_objective import (
    CorrectionGroup,
    CorrectionOptions,
    correction_loss,
    quality_scores,
)
from src.models.components.grpo import grpo_loss
from src.models.components.translation_reward import TranslationReward
from src.models.grpo_module import GRPOModule


class NoUsableFeedback(RuntimeError):
    """A complete rollout has no trusted update; skip without AdamW decay."""


class AnchoredGRPOModule(GRPOModule):
    """A hybrid objective, explicitly distinct from original GRPO and M2PO.

    Exact first-order gradient coefficients are computed on a detached score
    vector, then applied to fresh candidate graphs. This frees each graph after
    one backward, sharing each anchor score across all its preference pairs.
    """

    def __init__(self, correction: dict[str, Any] | None = None, **kwargs: Any) -> None:
        options = CorrectionOptions(**(correction or {}))
        if kwargs.get("temperature", 1.0) != 1.0 or kwargs.get("beta", 0.04) != 0:
            raise ValueError("Anchored objective requires temperature=1 and beta=0")
        super().__init__(**kwargs)
        self.save_hyperparameters("correction")
        self.correction = options
        self._skipped_rollouts = 0
        reward = self.reward_fn
        if not isinstance(reward, TranslationReward) or not reward.sac_enabled:
            raise ValueError("Anchored objective requires TranslationReward with SAC")
        if (
            reward.fusion.enabled
            or reward.polarity.enabled
            or reward.requires_consistency
            or reward.harness.enabled
        ):
            raise ValueError(
                "Use direct COMET x SAC: disable fusion, polarity, consistency and harness"
            )

    def _anchor_tokens(
        self, text: str, source_lang: str, target_lang: str
    ) -> list[int]:
        if self.hparams["language_codes"]:
            ids = multilingual_ids(
                self.tokenizer,
                [text],
                [source_lang],
                [target_lang],
                self.hparams["language_codes"],
                targets=True,
            )[0]
            limit = self.hparams["max_new_tokens"] + 1
        else:
            value = (
                text
                if self.is_encoder_decoder or self.hparams["use_chat_template"]
                else " " + text
            )
            ids = list(self.tokenizer(value, add_special_tokens=False)["input_ids"])
            if (
                self.is_encoder_decoder
                and self.hparams["forced_bos_token_id"] is not None
            ):
                ids.insert(0, self.hparams["forced_bos_token_id"])
            eos = self.tokenizer.eos_token_id
            if not ids or ids[-1] != eos:
                ids.append(eos)
            limit = self.hparams["max_new_tokens"]
        if not text.strip() or len(ids) > limit:
            raise ValueError(
                "Correction target is empty or exceeds its complete token budget"
            )
        return ids

    @torch.no_grad()
    def _rollout(self, batch: dict[str, list[str]]) -> dict[str, Any]:
        rollout = super()._rollout(batch)
        size, count = self.hparams["group_size"], len(batch["source"])
        components = rollout["reward_components"]
        quality = quality_scores(components["comet"], components["sac"])
        quality = [
            q if text.strip() else 0.0
            for q, text in zip(quality, rollout["predictions"], strict=True)
        ]
        # Pure reward constructor and objective must agree; no hidden fusion.
        if any(
            abs(q - r) > 1e-5
            for q, r in zip(quality, components["reward"], strict=True)
        ):
            raise ValueError(
                "Correction reward must be unnormalized direct COMET x SAC"
            )
        source_langs = batch.get("source_lang", [self.hparams["source_lang"]] * count)
        target_langs = batch.get("target_lang", [self.hparams["target_lang"]] * count)
        plans: list[tuple[int, list[int], int | None]] = []
        grpo_groups, needs_reference = [], []
        for group in range(count):
            indices = range(group * size, (group + 1) * size)
            rejected = [
                i
                for i in indices
                if components["sac"][i] < self.correction.rejected_sac_below
            ]
            if not rejected:
                grpo_groups.append(group)
                continue
            good = [
                i
                for i in indices
                if components["sac"][i] >= self.correction.anchor_sac_min
                and components["comet"][i] >= self.correction.anchor_comet_min
                and not rollout["capped"][i]
                and rollout["predictions"][i].strip()
            ]
            anchor = max(good, key=lambda i: quality[i]) if good else None
            plans.append((group, rejected, anchor))
            if anchor is None:
                needs_reference.append(group)
        # Verify only fallback references, never validation/test or a generated
        # "least-bad" target. Malformed/API failures propagate before any update.
        verified: dict[int, tuple[list[int], float]] = {}
        if needs_reference:
            texts = [batch["reference"][i] for i in needs_reference]
            refs = self._reward_components(
                texts,
                texts,
                [batch["source"][i] for i in needs_reference],
                source_langs=[source_langs[i] for i in needs_reference],
                target_langs=[target_langs[i] for i in needs_reference],
            )
            ref_quality = quality_scores(refs["comet"], refs["sac"])
            for j, i in enumerate(needs_reference):
                if (
                    refs["sac"][j] < self.correction.anchor_sac_min
                    or refs["comet"][j] < self.correction.anchor_comet_min
                ):
                    continue
                try:
                    tokens = self._anchor_tokens(
                        batch["reference"][i], source_langs[i], target_langs[i]
                    )
                except ValueError:
                    continue
                verified[i] = tokens, ref_quality[j]
        # Deduplicate exact completions within each source. Original GRPO rows
        # retain their independent sampling multiplicities through an index map.
        original_count = len(rollout["sequences"])
        inputs: list[dict[str, Tensor]] = []
        sequences: list[Tensor] = []
        masks: list[Tensor] = []
        mapping: list[int] = []
        keys: dict[tuple[int, tuple[int, ...]], int] = {}
        representatives: list[int] = []
        for i in range(original_count):
            end = int(rollout["mask"][i].nonzero()[-1]) + 1
            prefix = (
                1
                if self.is_encoder_decoder
                else rollout["inputs"]["input_ids"].shape[1]
            )
            seq = rollout["sequences"][i, : prefix + end]
            key = (i // size, tuple(seq.tolist()))
            if key not in keys:
                keys[key] = len(sequences)
                inputs.append({k: v[i] for k, v in rollout["inputs"].items()})
                sequences.append(seq)
                masks.append(rollout["mask"][i, :end])
                representatives.append(i)
            mapping.append(keys[key])
        corrections, reference_count, quarantined = [], 0, 0
        for group, rejected, anchor in plans:
            if anchor is None and group not in verified:
                quarantined += 1
                continue
            if anchor is not None:
                anchor_index, anchor_quality, provenance = (
                    mapping[anchor],
                    quality[anchor],
                    "candidate",
                )
            else:
                tokens, anchor_quality = verified[group]
                original = group * size
                source = {k: v[original] for k, v in rollout["inputs"].items()}
                if self.is_encoder_decoder:
                    seq = torch.tensor(
                        [self.generation_config.decoder_start_token_id, *tokens],
                        device=self.device,
                    )
                else:
                    seq = torch.cat(
                        [source["input_ids"], torch.tensor(tokens, device=self.device)]
                    )
                anchor_index, provenance = len(sequences), "training_reference"
                sequences.append(seq)
                mask = torch.ones(len(tokens), dtype=torch.bool, device=self.device)
                if (
                    self.hparams["language_codes"]
                    or self.hparams["forced_bos_token_id"] is not None
                ):
                    mask[0] = False
                masks.append(mask)
                inputs.append(source)
                reference_count += 1
            unique = {mapping[i]: i for i in rejected}
            candidates, weights = [], []
            for pool_index, original in unique.items():
                gap = min(
                    1.0,
                    max(
                        0.0,
                        anchor_quality
                        - quality[original]
                        - self.correction.min_quality_gap,
                    ),
                )
                if gap > 0:
                    candidates.append(pool_index)
                    weights.append(gap)
            corrections.append(
                CorrectionGroup(
                    anchor_index, tuple(candidates), tuple(weights), group, provenance
                )
            )
        # GRPO never sees the correction group's generated or reference targets.
        keep = torch.zeros_like(rollout["advantages"])
        for group in grpo_groups:
            keep[group * size : (group + 1) * size] = 1
        rollout["advantages"] = rollout["advantages"] * keep
        pool: dict[str, Any] = {
            "inputs": {k: torch.stack([row[k] for row in inputs]) for k in inputs[0]},
            "sequences": pad_sequence(
                sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id
            ),
            "mask": pad_sequence(masks, batch_first=True, padding_value=False),
        }
        # Original sampling already scored the same frozen rollout policy. Reuse
        # those values on update 1; only off-policy reference rows need a pass.
        initial = torch.zeros(pool["mask"].shape, device=self.device)
        width = min(initial.shape[1], rollout["old_logps"].shape[1])
        initial[: len(representatives), :width] = rollout["old_logps"][
            representatives, :width
        ]
        if reference_count:
            start = len(representatives)
            with self.phase_timing.measure(
                "correction/anchor_forward", self.device, samples=reference_count
            ):
                initial[start:] = self._logps(
                    self.policy,
                    {k: v[start:] for k, v in pool["inputs"].items()},
                    pool["sequences"][start:],
                    pool["mask"][start:],
                )
        pool["initial_scores"] = initial
        if not corrections and not rollout["advantages"].any():
            raise NoUsableFeedback("No verified correction or nonzero GRPO advantage")
        rollout.update(
            correction_pool=pool,
            correction_groups=corrections,
            correction_mapping=torch.tensor(mapping, device=self.device),
            correction_source_count=count,
            correction_routes={
                "grpo_groups": len(grpo_groups),
                "groups": len(corrections),
                "reference_anchors": reference_count,
                "quarantined_groups": quarantined,
                "unique_sequences": len(sequences),
                "original_sequences": original_count,
            },
        )
        return rollout

    def training_step(self, batch: dict[str, list[str]], batch_idx: int) -> Tensor:
        try:
            result = super().training_step(batch, batch_idx)
            self._skipped_rollouts = 0
            return result
        except NoUsableFeedback:
            self._skipped_rollouts += 1
            self.log(
                "train/correction/skipped_rollouts",
                float(self._skipped_rollouts),
                on_step=True,
                batch_size=len(batch["source"]),
            )
            if self._skipped_rollouts >= self.correction.max_consecutive_skips:
                raise RuntimeError(
                    "Repeated untrainable rollouts; inspect feedback/anchors before continuing"
                ) from None
            return torch.zeros((), device=self.device)

    def _backward_rollout(
        self,
        rollout: dict[str, Any],
        role: str = "translation",
        loss_weight: float = 1.0,
        backward_phase: str | None = None,
        backward_fn: Callable[[Tensor], None] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        pool = rollout["correction_pool"]
        # A fresh detached score pass permits arbitrary microbatch boundaries.
        # Reusing stale rollout scores after an optimizer update would be wrong.
        scores = pool.pop("initial_scores", None)
        if scores is None:
            with torch.no_grad(), self.phase_timing.measure(
                "correction/coefficient_forward",
                self.device,
                samples=len(pool["sequences"]),
            ):
                scores = self._logps(
                    self.policy, pool["inputs"], pool["sequences"], pool["mask"]
                )
        mapping = rollout["correction_mapping"]
        # Build the exact combined objective on a tiny token-score graph, rather
        # than retaining all transformer activations across preference pairs.
        with torch.enable_grad():
            leaf = scores.detach().requires_grad_(True)
            lengths = pool["mask"].sum(-1).clamp_min(1)
            means = (leaf * pool["mask"]).sum(-1) / lengths
            auxiliary, metrics = correction_loss(
                means,
                rollout["correction_groups"],
                self.correction,
                rollout["correction_source_count"],
            )
            width = leaf.shape[1]
            old = F.pad(
                rollout["old_logps"], (0, max(0, width - rollout["old_logps"].shape[1]))
            )[:, :width]
            original_mask = pool["mask"][mapping]
            pg, pg_metrics = grpo_loss(
                leaf[mapping],
                old,
                torch.zeros_like(old),
                rollout["advantages"],
                original_mask,
                self.hparams["clip_epsilon"],
                0,
            )
            objective = pg + auxiliary
            coefficients = torch.autograd.grad(objective, leaf)[0].detach()
        if not torch.isfinite(objective) or not torch.isfinite(coefficients).all():
            raise FloatingPointError(
                "Nonfinite correction objective; no update applied"
            )
        active = coefficients.ne(0).any(-1).nonzero().flatten()
        backward = backward_fn or self.manual_backward
        if not active.numel():
            backward(
                next(p for p in self.policy.parameters() if p.requires_grad).sum() * 0
            )
            metrics.update(pg_metrics)
            return objective.detach(), metrics
        selected = {
            "inputs": {k: v[active] for k, v in pool["inputs"].items()},
            "sequences": pool["sequences"][active],
            "mask": pool["mask"][active],
        }
        batches = self._candidate_batches(
            selected["inputs"], selected["sequences"], selected["mask"]
        )
        for batch in batches:
            with self.phase_timing.measure(
                "translation/current_forward",
                self.device,
                samples=len(batch.indices),
                input_tokens=batch.valid_tokens,
                padded_tokens=batch.padded_tokens,
            ):
                logps = self._score_batch(
                    self.policy, batch.inputs, batch.sequences, batch.mask
                )
            coeff = coefficients[active[batch.indices], : logps.shape[1]]
            chunk_loss = (logps * coeff).sum() * loss_weight
            if not torch.isfinite(chunk_loss):
                raise FloatingPointError(
                    "Nonfinite correction loss; no optimizer update applied"
                )
            with self.phase_timing.measure(
                backward_phase or "translation/backward",
                self.device,
                samples=len(batch.indices),
            ):
                backward(chunk_loss)
            del logps, chunk_loss
        metrics.update(pg_metrics)
        metrics.update(
            {
                f"correction/{k}": scores.new_tensor(v)
                for k, v in rollout["correction_routes"].items()
            }
        )
        metrics.update(
            scoring_microbatches=scores.new_tensor(len(batches)),
            scoring_candidates_per_graph_max=scores.new_tensor(
                max(len(batch.indices) for batch in batches)
            ),
            scoring_padded_tokens=scores.new_tensor(
                sum(batch.padded_tokens for batch in batches)
            ),
        )
        return objective.detach(), metrics
