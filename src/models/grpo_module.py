"""Lightning GRPO for Hugging Face encoder-decoder and causal translation policies."""

import csv
import hashlib
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from inspect import signature
from pathlib import Path
from typing import Any, cast

import lightning as L
import torch
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF
from torch import Tensor
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.data.components.multilingual_tokens import language_token_ids, multilingual_ids
from src.data.components.translation_prompt import translation_prompts
from src.models.components.adapters import (
    add_lora,
    disable_dropout,
    lora_enabled,
    quantized_load_options,
    validate_lora,
)
from src.models.components.candidate_batching import CandidateBatch, candidate_batches
from src.models.components.grpo import completion_mask, group_advantages, grpo_loss
from src.models.components.kl_control import KLController
from src.models.components.optim import TransferAdamW
from src.models.components.optim.factory import resolve_optimizer_factory
from src.models.components.optim.plateau import ValidationPlateau, plateau_options
from src.models.components.reward_diagnostics import (
    component_weights,
    reward_diagnostics,
)
from src.models.components.reward_harness import group_update_weights
from src.models.components.translation_reward import SAC_METRICS, TranslationReward
from src.utils.phase_timing import PhaseTiming
from src.utils.translation_telemetry import append_record, define_chart_axes, scalars


class GRPOModule(L.LightningModule):
    """Sample G translations per source, then reuse each rollout for K updates.

    Dropout stays disabled during scoring and optimization so the importance
    ratio describes the sampling policy. Sampling has full support: no top-k,
    nucleus filtering, repetition processors, or forced EOS are applied.
    Full tuning checkpoints reference weights; LoRA reconstructs a pinned frozen
    base and checkpoints adapters. Manual optimization supports one device per run.
    """

    needs_reference = True

    def __init__(
        self,
        model_name_or_path: str = "google/mt5-small",
        revision: str = "main",
        base_revision: str | None = None,
        tokenizer_name_or_path: str | None = None,
        source_lang: str = "English",
        target_lang: str = "Vietnamese",
        tokenizer_src_lang: str | None = None,
        language_codes: dict[str, str] | None = None,
        forced_bos_token_id: int | None = None,
        prompt_template: str = "Translate {source_lang} to {target_lang}: {source}",
        use_chat_template: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
        source_content_limit: int | None = None,
        strict_prompt_length: bool = False,
        max_source_length: int = 256,
        max_new_tokens: int = 128,
        group_size: int = 4,
        num_iterations: int = 2,
        temperature: float = 1.0,
        clip_epsilon: float = 0.2,
        beta: float = 0.04,
        kl_control: dict[str, Any] | None = None,
        learning_rate: float = 1e-6,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        lora: dict[str, Any] | None = None,
        load_dtype: str = "float32",
        quantization: str | None = None,
        gradient_checkpointing: bool = False,
        logprob_batch_size: int | None = None,
        logprob_token_budget: int | None = None,
        sort_candidates_by_length: bool = True,
        trim_logprob_padding: bool = True,
        warmup_steps: int = 0,
        optimizer: Callable[..., torch.optim.Optimizer] | dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        profiling: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
        save_predictions: bool = False,
        reward: Any = None,
        policy: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            *[
                name
                for name in signature(GRPOModule.__init__).parameters
                if name not in {"self", "policy", "tokenizer", "reward"}
            ]
        )
        self._optimizer_factory, self.hparams["optimizer"] = resolve_optimizer_factory(
            optimizer
        )
        self.phase_timing = PhaseTiming(**(profiling or {}))
        self.diagnostics = {
            "enabled": True,
            "every_n_updates": 1,
            "drift_every_n_updates": 20,
            "audit_prompt_lengths": True,
            "record_rollouts": True,
            **(diagnostics or {}),
        }
        for name in ("every_n_updates", "drift_every_n_updates"):
            value = self.diagnostics[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"diagnostics.{name} must be a positive integer")
        self._last_prompt_diagnostics: dict[str, float] = {}
        self._rollout_counter = 0
        self._loaded_optimizer_step = 0
        self._reward_updates_seen = 0
        self._plateau: ValidationPlateau | None = None
        self._restored_plateau: dict[str, Any] | None = None
        if lr_scheduler:
            plateau_options(lr_scheduler)
        self.kl_controller = KLController(beta, **(kl_control or {}))
        self._timing_role = "translation"
        if group_size < 2 or num_iterations < 1 or temperature <= 0:
            raise ValueError(
                "Need group_size >= 2, num_iterations >= 1, temperature > 0"
            )
        if (
            not 0 < clip_epsilon < 1
            or beta < 0
            or learning_rate <= 0
            or max_grad_norm <= 0
        ):
            raise ValueError(
                "Invalid clipping, KL coefficient, learning rate, or gradient norm"
            )
        if max_source_length < 1 or max_new_tokens < 1:
            raise ValueError("Token length limits must be positive")
        if source_content_limit is not None and (
            isinstance(source_content_limit, bool)
            or not isinstance(source_content_limit, int)
            or source_content_limit < 1
        ):
            raise ValueError("source_content_limit must be a positive integer or null")
        if load_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("load_dtype must be float32, bfloat16, or float16")
        if warmup_steps < 0:
            raise ValueError("Invalid warmup_steps")
        for name, value in (
            ("logprob_batch_size", logprob_batch_size),
            ("logprob_token_budget", logprob_token_budget),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or null")
        validate_lora(lora)
        self.uses_lora = lora_enabled(lora)
        load_options = quantized_load_options(quantization, load_dtype, self.uses_lora)
        if policy is not None and quantization is not None:
            raise ValueError(
                "Quantization must be configured while loading a model, not after injection"
            )
        if policy is None:
            model_revision = base_revision or revision
            config = AutoConfig.from_pretrained(
                model_name_or_path, revision=model_revision
            )
            factory = (
                AutoModelForSeq2SeqLM
                if config.is_encoder_decoder
                else AutoModelForCausalLM
            )
            policy = factory.from_pretrained(
                model_name_or_path,
                revision=model_revision,
                torch_dtype=getattr(torch, load_dtype),
                low_cpu_mem_usage=self.uses_lora,
                **load_options,
            )
        if self.uses_lora:
            assert lora is not None
            policy = add_lora(policy, lora, gradient_checkpointing)
        if gradient_checkpointing:
            policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            disable_dropout(policy)
        # Transformers dynamically mixes in generation; its public stubs are incomplete.
        self.policy: Any = policy
        self.hparams["base_revision"] = (
            getattr(policy.config, "_commit_hash", None) or base_revision or revision
        )
        self._loading_adapter_checkpoint = False
        if self.uses_lora:
            cast(Any, self.policy).peft_config["default"].revision = self.hparams[
                "base_revision"
            ]
        self.is_encoder_decoder = bool(policy.config.is_encoder_decoder)
        if forced_bos_token_id is not None and not self.is_encoder_decoder:
            raise ValueError(
                "forced_bos_token_id is supported only for encoder-decoder models"
            )
        self.tokenizer: Any = tokenizer or cast(Any, AutoTokenizer).from_pretrained(
            tokenizer_name_or_path or model_name_or_path,
            revision=revision
            if tokenizer_name_or_path
            else self.hparams["base_revision"],
        )
        if tokenizer_src_lang is not None:
            self.tokenizer.src_lang = tokenizer_src_lang
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs a PAD or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" if self.is_encoder_decoder else "left"
        if language_codes:
            if (
                not self.is_encoder_decoder
                or use_chat_template
                or prompt_template != "{source}"
                or forced_bos_token_id is not None
                or tokenizer_src_lang is not None
            ):
                raise ValueError(
                    "Per-example language codes require an encoder-decoder, raw {source}, and no static language/BOS or chat template"
                )
            language_token_ids(self.tokenizer, list(language_codes), language_codes)
        self.reference = (
            None
            if self.uses_lora or not self.needs_reference or beta == 0
            else deepcopy(self.policy).requires_grad_(False).eval()
        )
        self.warmup_steps = warmup_steps or (
            int((lora or {}).get("warmup_steps", 10)) if self.uses_lora else 0
        )
        self.reward_fn = reward if reward is not None else TranslationReward()
        self.automatic_optimization = False
        self._predictions: list[str] = []
        self._references: list[str] = []
        self._evaluation_rows: dict[str, list[dict[str, Any]]] = {}
        # Use a fresh config: inherited generation processors can invalidate GRPO ratios.
        original = self.policy.generation_config
        if original is None:
            raise ValueError("Policy must support Transformers text generation")
        self.generation_config = GenerationConfig(  # type: ignore[no-untyped-call]
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=original.eos_token_id,
            bos_token_id=original.bos_token_id,
            decoder_start_token_id=original.decoder_start_token_id,
            forced_bos_token_id=forced_bos_token_id,
        )

    def on_train_start(self) -> None:
        """Configure the plateau controller after Lightning restores optimizer state."""
        if self.hparams.get("lr_scheduler"):
            if self.hparams.get("cycle_stage"):
                raise ValueError("Plateau scheduling requires a direct SFT/RL stage")
            self._plateau = ValidationPlateau(
                self.trainer.optimizers[0],
                self.hparams["learning_rate"],
                self.hparams["lr_scheduler"],
                self._restored_plateau,
            )
            schedule_dir = Path(self.trainer.default_root_dir) / "diagnostics"
            schedule_dir.mkdir(parents=True, exist_ok=True)
            (schedule_dir / "lr_schedule.json").write_text(
                json.dumps(
                    {
                        "optimizer": type(self.trainer.optimizers[0]).__name__,
                        "actual_lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                        "global_step": self.global_step,
                        "resumed_schedule": self._plateau.resumed,
                        "config": self._plateau.config,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    def on_fit_start(self) -> None:
        """Fail explicitly for unsupported manual-optimization trainer settings."""
        if isinstance(self.reward_fn, TranslationReward):
            self.reward_fn.harness.check_ready()
            if self.reward_fn.sac is not None:
                self.reward_fn.sac.check_ready()
        if self.trainer.world_size != 1:
            raise ValueError(
                "GRPO currently supports one device; distributed rollouts are not implemented"
            )
        if self.trainer.accumulate_grad_batches != 1 or self.trainer.gradient_clip_val:
            raise ValueError(
                "Use model.max_grad_norm; trainer accumulation/clipping is unsupported"
            )
        self.phase_timing.configure(
            Path(self.trainer.default_root_dir) / "phase_timing"
        )
        trainable = {
            name: p.numel()
            for name, p in self.policy.named_parameters()
            if p.requires_grad
        }
        total = sum(p.numel() for p in self.policy.parameters())
        audit = {
            "total_parameters": total,
            "trainable_parameters": sum(trainable.values()),
            "trainable_fraction": sum(trainable.values()) / total,
            "lora_enabled": self.uses_lora,
            "backbone_frozen": self.uses_lora
            and all("lora_" in name for name in trainable),
            "trainable_tensors": trainable,
        }
        output = Path(self.trainer.default_root_dir) / "diagnostics"
        output.mkdir(parents=True, exist_ok=True)
        (output / "trainability.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        for logger in self.trainer.loggers:
            experiment = getattr(logger, "experiment", None)
            if callable(getattr(experiment, "define_metric", None)):
                define_chart_axes(experiment)

    def _note_augmentation(self, batch: dict[str, list[str]]) -> None:
        if "augmentation_seconds" in batch:
            self.phase_timing.record(
                "data/augmentation",
                sum(float(value) for value in batch["augmentation_seconds"]),
                samples=len(batch["source"]),
            )

    def _format_prompts(
        self,
        sources: list[str],
        source_langs: list[str] | None = None,
        target_langs: list[str] | None = None,
    ) -> list[str]:
        source_langs = source_langs or [self.hparams["source_lang"]] * len(sources)
        target_langs = target_langs or [self.hparams["target_lang"]] * len(sources)
        if len(source_langs) != len(sources) or len(target_langs) != len(sources):
            raise ValueError("Every source needs source/target language metadata")
        if self.hparams["forced_bos_token_id"] is not None and any(
            lang != self.hparams["target_lang"] for lang in target_langs
        ):
            raise ValueError(
                "Fixed forced_bos_token_id conflicts with a reverse translation direction"
            )
        if self.hparams["tokenizer_src_lang"] is not None and any(
            lang != self.hparams["source_lang"] for lang in source_langs
        ):
            raise ValueError(
                "Fixed tokenizer_src_lang conflicts with a reverse translation direction"
            )
        content_limit = self.hparams["source_content_limit"]
        if content_limit is not None:
            ids = self.tokenizer(sources, add_special_tokens=False, truncation=False)[
                "input_ids"
            ]
            if any(len(value) > content_limit for value in ids):
                raise ValueError(
                    "Source exceeds source_content_limit; select complete token-bounded pairs"
                )
        if self.hparams["language_codes"]:
            return list(sources)
        return translation_prompts(
            self.tokenizer,
            sources,
            source_langs,
            target_langs,
            self.hparams["prompt_template"],
            self.hparams["use_chat_template"],
            self.hparams["chat_template_kwargs"],
        )

    def _encode(
        self,
        sources: list[str],
        source_langs: list[str] | None = None,
        target_langs: list[str] | None = None,
    ) -> dict[str, Tensor]:
        prompts = self._format_prompts(sources, source_langs, target_langs)
        return (
            self._tokenize_prompts(prompts, source_langs, target_langs)
            if self.hparams["language_codes"]
            else self._tokenize_prompts(prompts)
        )

    def _tokenize_prompts(
        self,
        prompts: list[str],
        source_langs: list[str] | None = None,
        target_langs: list[str] | None = None,
    ) -> dict[str, Tensor]:
        """Tokenize complete role prompts without adding a translation instruction."""
        strict = self.hparams["strict_prompt_length"]
        if self.hparams["language_codes"]:
            source_langs = source_langs or [self.hparams["source_lang"]] * len(prompts)
            target_langs = target_langs or [self.hparams["target_lang"]] * len(prompts)
            ids = multilingual_ids(
                self.tokenizer,
                prompts,
                source_langs,
                target_langs,
                self.hparams["language_codes"],
            )
            lengths = [len(value) for value in ids]
            over = [length > self.hparams["max_source_length"] for length in lengths]
            self._last_prompt_diagnostics = {
                "prompt/full_tokens_mean": sum(lengths) / len(lengths),
                "prompt/truncated_fraction": sum(over) / len(lengths),
            }
            if any(over):
                if strict:
                    raise ValueError(
                        "Complete multilingual source exceeds max_source_length; re-filter pairs"
                    )
                ids = multilingual_ids(
                    self.tokenizer,
                    prompts,
                    source_langs,
                    target_langs,
                    self.hparams["language_codes"],
                    truncation=True,
                    max_length=self.hparams["max_source_length"],
                )
            return dict(
                self.tokenizer.pad(
                    {"input_ids": ids}, padding=True, return_tensors="pt"
                ).to(self.device)
            )
        token_options = (
            {"add_special_tokens": False} if self.hparams["use_chat_template"] else {}
        )
        length_options = (
            {} if strict else {"max_length": self.hparams["max_source_length"]}
        )
        if strict or (
            self.diagnostics["enabled"] and self.diagnostics["audit_prompt_lengths"]
        ):
            full = self.tokenizer(
                prompts,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                **token_options,
            )
            lengths = [len(ids) for ids in full["input_ids"]]
            if strict and any(
                length > self.hparams["max_source_length"] for length in lengths
            ):
                raise ValueError(
                    "Complete translation prompt exceeds max_source_length; "
                    "re-filter pairs with the current tokenizer/instruction instead of cutting chat roles"
                )
            self._last_prompt_diagnostics = {
                "prompt/full_tokens_mean": sum(lengths) / len(lengths),
                "prompt/truncated_fraction": sum(
                    length > self.hparams["max_source_length"] for length in lengths
                )
                / len(lengths),
            }
        return dict(
            self.tokenizer(
                prompts,
                padding=True,
                truncation=not strict,
                **length_options,
                return_tensors="pt",
                return_token_type_ids=False,
                **token_options,
            ).to(self.device)
        )

    def _logps(
        self,
        model: Any,
        inputs: dict[str, Tensor],
        sequences: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Score in length batches and restore rollout order, principally without grad.

        Training uses _backward_rollout instead, freeing each graph immediately.
        Unscored positions are zero regardless of the batch's padding shape.
        """
        batches = self._candidate_batches(inputs, sequences, mask)
        scores = [
            F.pad(
                self._score_batch(
                    model, batch.inputs, batch.sequences, batch.mask
                ).masked_fill(~batch.mask, 0),
                (0, mask.shape[1] - batch.mask.shape[1]),
            )
            for batch in batches
        ]
        inverse = torch.cat([batch.indices for batch in batches]).argsort()
        return torch.cat(scores).index_select(0, inverse)

    def _candidate_batches(
        self,
        inputs: dict[str, Tensor],
        sequences: Tensor,
        mask: Tensor,
    ) -> list[CandidateBatch]:
        prefix = 1 if self.is_encoder_decoder else inputs["input_ids"].shape[1]
        context = completion_mask(
            sequences[:, prefix:], self.generation_config.eos_token_id
        )
        return candidate_batches(
            inputs,
            sequences,
            mask,
            context,
            self.is_encoder_decoder,
            self.hparams["logprob_batch_size"],
            self.hparams["logprob_token_budget"],
            self.hparams["sort_candidates_by_length"],
            self.hparams["trim_logprob_padding"],
        )

    def _score_batch(
        self,
        model: Any,
        inputs: dict[str, Tensor],
        sequences: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """One vectorized candidate forward; all rows share this computation graph."""
        if self.is_encoder_decoder:
            targets = sequences[:, 1:]
            context_mask = completion_mask(targets, self.generation_config.eos_token_id)
            decoder_attention = torch.cat(
                [torch.ones_like(context_mask[:, :1]), context_mask[:, :-1]], dim=1
            )
            logits = model(
                **inputs,
                decoder_input_ids=sequences[:, :-1],
                decoder_attention_mask=decoder_attention,
                use_cache=False,
            ).logits
        else:
            prefix = inputs["input_ids"].shape[1]
            targets = sequences[:, prefix:]
            attention = torch.cat([inputs["attention_mask"], mask], dim=1)
            extra: dict[str, Any] = {}
            base_forward = (
                model.get_base_model().forward if self.uses_lora else model.forward
            )
            if "position_ids" in signature(base_forward).parameters:
                extra["position_ids"] = (attention.long().cumsum(-1) - 1).clamp_min(0)
            keep_completion_logits = (
                "logits_to_keep" in signature(base_forward).parameters
            )
            if keep_completion_logits:
                extra["logits_to_keep"] = targets.shape[1] + 1
            output_logits = model(
                input_ids=sequences, attention_mask=attention, use_cache=False, **extra
            ).logits
            logits = (
                output_logits[:, :-1]
                if keep_completion_logits
                else output_logits[:, prefix - 1 : -1]
            )
        # FP32 softmax avoids mixed-precision underflow in likelihood ratios and KL.
        logits = logits.float() / self.hparams["temperature"]
        return cast(
            Tensor, logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        )

    def _backward_rollout(
        self,
        rollout: dict[str, Any],
        role: str = "translation",
        loss_weight: float = 1.0,
        backward_phase: str | None = None,
        backward_fn: Callable[[Tensor], None] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Accumulate the exact full-rollout gradient with one graph live at a time.

        Advantages, old scores and reference scores are fixed across all chunks.
        The caller zeroes, clips and steps once per optimizer update, including
        when multiple self-play roles contribute to that same update.
        """
        batches = self._candidate_batches(
            rollout["inputs"], rollout["sequences"], rollout["mask"]
        )
        count = len(rollout["sequences"])
        tokens = max(1, sum(batch.scored_tokens for batch in batches))
        backward = backward_fn or self.manual_backward
        total_loss = torch.zeros((), device=self.device)
        metrics: dict[str, Tensor] = {}
        token_metrics = {
            "kl",
            "clip_fraction",
            "ratio_mean",
            "approx_kl_to_old",
            "sampled_token_nll",
        }
        for batch in batches:
            with self.phase_timing.measure(
                f"{role}/current_forward",
                self.device,
                samples=len(batch.indices),
                input_tokens=batch.valid_tokens,
                padded_tokens=batch.padded_tokens,
            ):
                logps = self._score_batch(
                    self.policy, batch.inputs, batch.sequences, batch.mask
                )
            width = batch.mask.shape[1]
            loss, chunk_metrics = grpo_loss(
                logps,
                rollout["old_logps"][batch.indices, :width],
                rollout["reference_logps"][batch.indices, :width],
                rollout["advantages"][batch.indices],
                batch.mask,
                self.hparams["clip_epsilon"],
                self.kl_controller.value,
            )
            fraction = len(batch.indices) / count
            with self.phase_timing.measure(
                backward_phase or f"{role}/backward",
                self.device,
                samples=len(batch.indices),
            ):
                backward(loss * (fraction * loss_weight))
            total_loss += loss.detach() * fraction
            for name, value in chunk_metrics.items():
                weight = (
                    batch.scored_tokens / tokens if name in token_metrics else fraction
                )
                weighted = value.detach() * weight
                metrics[name] = (
                    metrics[name] + weighted if name in metrics else weighted
                )
            # Do not keep the most recent graph alive into the following forward.
            del loss, logps, chunk_metrics, value
        # Check once before the caller can clip/step; avoid one CUDA-host barrier
        # per microbatch in normal (non-synchronized profiling) training.
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite GRPO loss; no optimizer update applied"
            )
        padded = sum(batch.padded_tokens for batch in batches)
        valid = sum(batch.valid_tokens for batch in batches)
        metrics.update(
            {
                "scoring_microbatches": torch.tensor(len(batches), device=self.device),
                "scoring_candidates_per_graph_max": torch.tensor(
                    max(len(batch.indices) for batch in batches), device=self.device
                ),
                "scoring_padded_tokens": torch.tensor(padded, device=self.device),
                "scoring_padding_fraction": torch.tensor(
                    1 - valid / padded, device=self.device
                ),
            }
        )
        return total_loss, metrics

    @torch.no_grad()
    def _reference_logps(
        self, inputs: dict[str, Tensor], sequences: Tensor, mask: Tensor
    ) -> Tensor:
        """Score the original frozen policy without allocating a second LoRA base."""
        if self.uses_lora:
            with self.policy.disable_adapter():
                return self._logps(self.policy, inputs, sequences, mask)
        assert self.reference is not None
        return self._logps(self.reference, inputs, sequences, mask)

    def _prepare_gradient_scoring(self) -> None:
        """Activate checkpoint recomputation while keeping all dropout disabled."""
        self.policy.train(bool(self.hparams["gradient_checkpointing"]))

    def _optimizer_step(self, optimizer: Any) -> None:
        """Apply resumable linear warmup using Lightning's optimizer-step counter."""
        if self.warmup_steps and self._plateau is None:
            fraction = min(1.0, (self.global_step + 1) / self.warmup_steps)
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
                group["lr"] = group["initial_lr"] * fraction
        optimizer.step()

    def _generate_policy(
        self, inputs: dict[str, Tensor], config: GenerationConfig, **kwargs: Any
    ) -> Any:
        """Prevent Hub decoding defaults from changing the policy being scored.

        Transformers >=4.50 can replace *explicit* global-default values such as
        top_p=1 and temperature=1 with a model's generation_config. Disable that
        merge, including through PEFT wrappers; retain compatibility with 4.45.
        """
        base = self.policy.get_base_model() if self.uses_lora else self.policy
        if "use_model_defaults" in signature(base.generate).parameters:
            kwargs["use_model_defaults"] = False
        return self.policy.generate(**inputs, generation_config=config, **kwargs)

    def _language_generation_prefix(
        self, target_langs: list[str] | None, count: int, groups: int = 1
    ) -> dict[str, Tensor]:
        """Seed each NLLB decoder with [decoder-start, its own target-language token]."""
        codes = self.hparams["language_codes"]
        if not codes:
            return {}
        names = target_langs or [self.hparams["target_lang"]] * count
        if len(names) != count:
            raise ValueError("Every generated source needs a target language")
        start = self.generation_config.decoder_start_token_id
        if start is None:
            raise ValueError("Multilingual generation requires decoder_start_token_id")
        ids = language_token_ids(self.tokenizer, names, codes)
        prefix = torch.tensor(
            [[start, value] for value in ids], dtype=torch.long, device=self.device
        )
        return {"decoder_input_ids": prefix.repeat_interleave(groups, dim=0)}

    @torch.no_grad()
    def _sample_prompts(
        self,
        prompts: list[str],
        group_size: int,
        max_new_tokens: int | None = None,
        source_langs: list[str] | None = None,
        target_langs: list[str] | None = None,
    ) -> dict[str, Any]:
        """Collect detached rollout scores for arbitrary proposer or solver prompts."""
        self.policy.eval()
        if self.reference is not None:
            self.reference.eval()
        encoded = (
            self._tokenize_prompts(prompts, source_langs, target_langs)
            if self.hparams["language_codes"]
            else self._tokenize_prompts(prompts)
        )
        inputs = {k: v.repeat_interleave(group_size, dim=0) for k, v in encoded.items()}
        config = deepcopy(self.generation_config)
        if max_new_tokens is not None:
            config.max_new_tokens = max_new_tokens
        with self.phase_timing.measure(
            f"{self._timing_role}/generation",
            self.device,
            samples=inputs["input_ids"].shape[0],
            input_tokens=int(inputs["attention_mask"].sum()),
        ) as generation_work:
            generation_kwargs = self._language_generation_prefix(
                target_langs, len(prompts), group_size
            )
            sequences = self._generate_policy(inputs, config, **generation_kwargs)
            prefix = 1 if self.is_encoder_decoder else inputs["input_ids"].shape[1]
            completions = sequences[:, prefix:]
            mask = completion_mask(completions, self.generation_config.eos_token_id)
            if (
                self.hparams["forced_bos_token_id"] is not None
                or self.hparams["language_codes"]
            ):
                mask[:, 0] = False
            generation_work["output_tokens"] = int(mask.sum())
        # A forced language token is deterministic and must not contribute to the objective.
        if (
            self.hparams["forced_bos_token_id"] is not None
            or self.hparams["language_codes"]
        ):
            mask[:, 0] = False
        predictions = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        with self.phase_timing.measure(
            f"{self._timing_role}/old_policy_forward",
            self.device,
            samples=len(sequences),
            input_tokens=int(inputs["attention_mask"].sum() + mask.sum()),
        ):
            old_logps = self._logps(self.policy, inputs, sequences, mask)
        reference_logps = torch.zeros_like(old_logps)
        if self.kl_controller.value > 0:
            with self.phase_timing.measure(
                f"{self._timing_role}/reference_forward",
                self.device,
                samples=len(sequences),
                input_tokens=int(inputs["attention_mask"].sum() + mask.sum()),
            ):
                reference_logps = self._reference_logps(inputs, sequences, mask)
        return {
            "inputs": inputs,
            "sequences": sequences,
            "mask": mask,
            "predictions": predictions,
            "old_logps": old_logps,
            "reference_logps": reference_logps,
            "prompt_diagnostics": dict(self._last_prompt_diagnostics),
            "capped": (
                completions.shape[1] - int(bool(self.hparams["language_codes"]))
                >= config.max_new_tokens
            )
            & ~(
                (
                    completions.unsqueeze(-1)
                    == torch.as_tensor(
                        config.eos_token_id if config.eos_token_id is not None else [],
                        device=self.device,
                    ).reshape(-1)
                ).any(-1)
                & mask
            ).any(-1),
        }

    @torch.no_grad()
    def _rollout(self, batch: dict[str, list[str]]) -> dict[str, Any]:
        partners = self._paired_reward_indices(batch, self.hparams["group_size"])
        prompts = self._format_prompts(
            batch["source"], batch.get("source_lang"), batch.get("target_lang")
        )
        groups = self.hparams["group_size"]
        language_options: dict[str, Any] = (
            {
                "source_langs": batch.get("source_lang"),
                "target_langs": batch.get("target_lang"),
            }
            if self.hparams["language_codes"]
            else {}
        )
        rollout = self._sample_prompts(prompts, groups, **language_options)
        references = [r for r in batch["reference"] for _ in range(groups)]
        sources = [
            source
            for source in batch.get("clean_source", batch["source"])
            for _ in range(groups)
        ]
        components = self._reward_components(
            rollout["predictions"],
            references,
            sources,
            source_langs=[
                value
                for value in batch.get(
                    "source_lang", [self.hparams["source_lang"]] * len(batch["source"])
                )
                for _ in range(groups)
            ],
            target_langs=[
                value
                for value in batch.get(
                    "target_lang", [self.hparams["target_lang"]] * len(batch["source"])
                )
                for _ in range(groups)
            ],
            partner_indices=partners,
            candidate_group_size=groups,
        )
        rewards = torch.tensor(
            components["reward"],
            device=self.device,
            dtype=torch.float32,
        )
        if (
            rewards.shape != (len(rollout["predictions"]),)
            or not torch.isfinite(rewards).all()
        ):
            raise ValueError("Reward must return one finite scalar per candidate")
        weights = torch.tensor(
            group_update_weights(components, groups, len(rewards)), device=self.device
        )
        return {
            **rollout,
            "advantages": group_advantages(rewards, groups) * weights,
            "update_weights": weights,
            "rewards": rewards,
            "reward_components": components,
        }

    def _paired_reward_indices(
        self, batch: dict[str, list[str]], groups: int
    ) -> list[int] | None:
        if (
            isinstance(self.reward_fn, TranslationReward)
            and self.reward_fn.sac_enabled
            and self.reward_fn.consistency_mode == "symmetric_endpoints"
        ):
            from src.models.components.symmetric_alignment import (
                paired_candidate_indices,
            )

            return paired_candidate_indices(batch, groups)
        return None

    def _reward_progress(self) -> int:
        stage = self.hparams.get("cycle_stage")
        if stage == "sft" or (stage is None and not self.needs_reference):
            return 0
        if stage == "rl":
            return max(
                int(self.hparams.get("stage_update_offset", 0)) + self.global_step,
                self._reward_updates_seen,
            )
        return max(
            self.global_step, self._loaded_optimizer_step, self._reward_updates_seen
        )

    def _reward_components(
        self,
        predictions: list[str],
        references: list[str],
        sources: list[str],
        source_langs: list[str] | None = None,
        target_langs: list[str] | None = None,
        apply_harness: bool = True,
        partner_indices: list[int] | None = None,
        candidate_group_size: int = 1,
    ) -> dict[str, list[float]]:
        if isinstance(self.reward_fn, TranslationReward):
            reconstructions = None
            if (
                self.reward_fn.sac_enabled
                and self.reward_fn.consistency_mode == "round_trip"
                and self.reward_fn.requires_consistency
            ):
                if source_langs is None or target_langs is None:
                    raise ValueError(
                        "Round-trip translation requires both language labels"
                    )
                reconstructions = []
                size = self.reward_fn.round_trip_batch_size
                with self.phase_timing.measure(
                    "reward/round_trip_generation",
                    self.device,
                    samples=len(predictions),
                ):
                    for offset in range(0, len(predictions), size):
                        end = offset + size
                        reconstructions.extend(
                            self.translate(
                                predictions[offset:end],
                                source_langs=target_langs[offset:end],
                                target_langs=source_langs[offset:end],
                                max_new_tokens=self.reward_fn.round_trip_max_new_tokens,
                            )
                        )
            result = self.reward_fn.components(
                predictions,
                references,
                sources=sources,
                source_langs=source_langs,
                target_langs=target_langs,
                apply_harness=apply_harness,
                reconstructions=reconstructions,
                partner_indices=partner_indices,
                optimizer_step=self._reward_progress(),
                candidate_group_size=candidate_group_size,
            )
            for phase, seconds in self.reward_fn.last_timings.items():
                self.phase_timing.record(
                    f"reward/{phase}",
                    seconds,
                    samples=len(predictions),
                    input_tokens=getattr(self.reward_fn.comet, "last_input_tokens", 0)
                    if phase == "cometkiwi_forward"
                    else 0,
                )
            return result
        rewards = (
            self.reward_fn(predictions, references, sources=sources)
            if getattr(self.reward_fn, "requires_sources", False)
            else self.reward_fn(predictions, references)
        )
        return {"reward": rewards}

    def training_step(self, batch: dict[str, list[str]], batch_idx: int) -> Tensor:
        with self.phase_timing.measure(
            "translation/batch_total", self.device, samples=len(batch["source"])
        ):
            return self._translation_step(batch, batch_idx)

    def _translation_step(self, batch: dict[str, list[str]], batch_idx: int) -> Tensor:
        """Run clipped policy updates against fixed rollout and reference scores."""
        rollout = self._rollout(batch)
        self._rollout_counter += 1
        if (
            isinstance(self.reward_fn, TranslationReward)
            and self.reward_fn.harness.enabled
        ):
            failed = rollout["reward_components"]["constraint_failure"]
            groups = self.hparams["group_size"]
            repair_groups = [
                i
                for i in range(len(batch["source"]))
                if all(v == 1 for v in failed[i * groups : (i + 1) * groups])
            ]
            if repair_groups:
                # Queue identifiers, never treat a synthetic failure as a verified target.
                append_record(
                    Path(self.trainer.default_root_dir)
                    / "diagnostics/repair_queue.jsonl",
                    {
                        "global_step": self.global_step,
                        "batch_idx": batch_idx,
                        "groups": [
                            {
                                "group": i,
                                "source_sha256": hashlib.sha256(
                                    batch.get("clean_source", batch["source"])[
                                        i
                                    ].encode()
                                ).hexdigest(),
                                "direction": batch.get(
                                    "direction", ["forward"] * len(batch["source"])
                                )[i],
                            }
                            for i in repair_groups
                        ],
                        "action": "verified_correction_required",
                    },
                )
            append_record(
                Path(self.trainer.default_root_dir) / "diagnostics/criteria.jsonl",
                {
                    "global_step": self.global_step,
                    "criteria": self.reward_fn.harness.last_evidence,
                },
            )
        rollout_metrics: dict[str, float] = {}
        if self.diagnostics["enabled"]:
            rollout_metrics = reward_diagnostics(
                rollout["rewards"],
                rollout["reward_components"],
                self.hparams["group_size"],
                rollout["predictions"],
                batch.get("direction"),
                component_weights(self.reward_fn),
            )
            rollout_metrics.update(rollout["prompt_diagnostics"])
            if (
                isinstance(self.reward_fn, TranslationReward)
                and self.reward_fn.sac is not None
            ):
                rollout_metrics.update(
                    {
                        f"reward/sac_{name}": value
                        for name, value in self.reward_fn.sac.last_metrics.items()
                    }
                )
            rollout_metrics["rollout/capped_fraction"] = float(
                rollout["capped"].float().mean()
            )
            if self.diagnostics["record_rollouts"]:
                append_record(
                    Path(self.trainer.default_root_dir) / "diagnostics/rollouts.jsonl",
                    {
                        "rollout": self._rollout_counter,
                        "global_step": self.global_step,
                        "group_size": self.hparams["group_size"],
                        "directions": batch.get("direction"),
                        "components": rollout["reward_components"],
                        "weights": component_weights(self.reward_fn),
                        "rewards": rollout["rewards"].cpu().tolist(),
                        "completion_lengths": rollout["mask"].sum(-1).cpu().tolist(),
                        "capped": rollout["capped"].cpu().tolist(),
                    },
                )
        self._note_augmentation(batch)
        optimizer = cast(Any, self.optimizers())
        for iteration in range(self.hparams["num_iterations"]):
            # Manual optimization can otherwise overshoot odd max_steps when K=2.
            if (
                self.trainer.max_steps >= 0
                and self.global_step >= self.trainer.max_steps
            ):
                break
            beta_used = self.kl_controller.value
            optimizer.zero_grad()
            self._prepare_gradient_scoring()
            loss, metrics = self._backward_rollout(rollout)
            grad_norm = torch.stack(
                [
                    p.grad.detach().float().norm()
                    for p in self.policy.parameters()
                    if p.requires_grad and p.grad is not None
                ]
            ).norm()
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    "Non-finite policy gradients; optimizer update skipped"
                )
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.hparams["max_grad_norm"],
                gradient_clip_algorithm="norm",
            )
            with self.phase_timing.measure("translation/optimizer", self.device):
                self._optimizer_step(optimizer)
            self._reward_updates_seen = self._reward_progress()
            if self.kl_controller.enabled:
                self.kl_controller.observe(float(metrics["kl"]))
            metrics.update(
                kl_beta=torch.tensor(beta_used, device=self.device),
                kl_beta_next=torch.tensor(self.kl_controller.value, device=self.device),
            )
            if (
                self.diagnostics["enabled"]
                and self.global_step % self.diagnostics["every_n_updates"] == 0
            ):
                diagnostic_metrics: dict[str, Any] = {
                    **rollout_metrics,
                    **{f"grpo/{key}": value for key, value in metrics.items()},
                    "grpo/loss": loss,
                    "optimizer/gradient_norm": grad_norm,
                    "optimizer/gradient_clip_scale": (
                        self.hparams["max_grad_norm"] / (grad_norm + 1e-6)
                    ).clamp(max=1),
                    "optimizer/lr": optimizer.param_groups[0]["lr"],
                    "optimizer/warmup_fraction": min(
                        1, self.global_step / self.warmup_steps
                    )
                    if self.warmup_steps
                    else 1,
                    "rollout/id": self._rollout_counter,
                    "rollout/reuse_iteration": iteration,
                }
                raw_optimizer = self.trainer.optimizers[0]
                if (
                    isinstance(raw_optimizer, TransferAdamW)
                    and self.global_step % self.diagnostics["drift_every_n_updates"]
                    == 0
                ):
                    diagnostic_metrics.update(
                        {
                            f"optimizer/transfer/{key}": value
                            for key, value in raw_optimizer.drift_metrics().items()
                        }
                    )
                values = scalars(diagnostic_metrics)
                append_record(
                    Path(self.trainer.default_root_dir) / "diagnostics/updates.jsonl",
                    {
                        "global_step": self.global_step,
                        "epoch": self.current_epoch,
                        "metrics": values,
                    },
                )
                for logger in self.trainer.loggers:
                    logger.log_metrics(values, step=self.global_step)
        metrics.update(loss=loss.detach(), reward=rollout["rewards"].mean())
        for name, values in rollout["reward_components"].items():
            if name != "reward":
                metrics[f"reward_{name}"] = torch.tensor(
                    values, device=self.device
                ).mean()
        grouped_rewards = rollout["rewards"].reshape(-1, self.hparams["group_size"])
        metrics["reward_std"] = grouped_rewards.std(-1, correction=0).mean()
        metrics["zero_variance_groups"] = (
            (grouped_rewards.std(-1, correction=0) == 0).float().mean()
        )
        self.log_dict(
            {f"train/{k}": v for k, v in metrics.items()},
            batch_size=len(batch["source"]),
            on_step=True,
            on_epoch=True,
        )
        return loss.detach()

    @torch.no_grad()
    def translate(
        self,
        sources: list[str],
        source_langs: list[str] | None = None,
        target_langs: list[str] | None = None,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Decode one greedy translation per source using the current policy."""
        was_training = self.policy.training
        self.policy.eval()
        try:
            inputs = self._encode(sources, source_langs, target_langs)
            config = deepcopy(self.generation_config)
            if max_new_tokens is not None:
                config.max_new_tokens = max_new_tokens
            config.do_sample = False
            config.temperature = 1.0
            config.top_k = 50  # Restore the default for a valid non-sampling config.
            with self.phase_timing.measure(
                "evaluation/generation",
                self.device,
                samples=len(sources),
                input_tokens=int(inputs["attention_mask"].sum()),
            ) as work:
                sequences = self._generate_policy(
                    inputs,
                    config,
                    **self._language_generation_prefix(target_langs, len(sources)),
                )
                prefix = 1 if self.is_encoder_decoder else inputs["input_ids"].shape[1]
                work["output_tokens"] = int(
                    completion_mask(
                        sequences[:, prefix:], self.generation_config.eos_token_id
                    ).sum()
                )
            return list(
                self.tokenizer.batch_decode(
                    sequences[:, prefix:], skip_special_tokens=True
                )
            )
        finally:
            self.policy.train(was_training)

    def _evaluation_step(self, batch: dict[str, list[str]], stage: str) -> None:
        partners = self._paired_reward_indices(batch, 1)
        predictions = (
            self.translate(
                batch["source"],
                source_langs=batch["source_lang"],
                target_langs=batch["target_lang"],
            )
            if "source_lang" in batch
            else self.translate(batch["source"])
        )
        sources = batch.get("clean_source", batch["source"])
        components = self._reward_components(
            predictions,
            batch["reference"],
            sources,
            source_langs=batch.get(
                "source_lang", [self.hparams["source_lang"]] * len(sources)
            ),
            target_langs=batch.get(
                "target_lang", [self.hparams["target_lang"]] * len(sources)
            ),
            apply_harness=False,
            partner_indices=partners,
        )
        rewards = components["reward"]
        for index, (source, prediction, reference) in enumerate(
            zip(sources, predictions, batch["reference"], strict=True)
        ):
            direction = batch.get("direction", ["forward"] * len(sources))[index]
            row: dict[str, Any] = {
                "example_index": len(self._predictions) + index,
                "direction": direction,
                "source": source,
                "prediction": prediction,
                "reference": reference,
                "reward": rewards[index],
            }
            if "comet" in components:
                row["comet"] = components["comet"][index]
            for name in SAC_METRICS:
                if name in components:
                    row[name] = components[name][index]
            self._evaluation_rows.setdefault(direction, []).append(row)
            if self.hparams["save_predictions"]:
                append_record(
                    Path(self.trainer.default_root_dir)
                    / "translation_metrics"
                    / f"{stage}_predictions.jsonl",
                    row,
                )
        self._predictions.extend(predictions)
        self._references.extend(batch["reference"])
        self.log(
            f"{stage}/reward",
            sum(rewards) / len(rewards),
            batch_size=len(rewards),
            on_epoch=True,
        )

    def validation_step(self, batch: dict[str, list[str]], batch_idx: int) -> None:
        """Evaluate deterministic translations."""
        self._evaluation_step(batch, "val")

    def test_step(self, batch: dict[str, list[str]], batch_idx: int) -> None:
        """Evaluate held-out translations."""
        self._evaluation_step(batch, "test")

    def on_validation_epoch_start(self) -> None:
        """Reset corpus buffers, including before sanity validation."""
        self._predictions.clear()
        self._references.clear()
        self._evaluation_rows.clear()
        if self.hparams["save_predictions"]:
            stage = "test" if self.trainer.testing else "val"
            directory = Path(self.trainer.default_root_dir) / "translation_metrics"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{stage}_predictions.jsonl").write_text("", encoding="utf-8")
        self.phase_timing.configure(
            Path(self.trainer.default_root_dir) / "phase_timing"
        )

    def on_test_epoch_start(self) -> None:
        """Reset test corpus buffers."""
        self.on_validation_epoch_start()

    def _evaluation_end(self, stage: str) -> None:
        if self.trainer.world_size != 1:
            raise ValueError("Corpus evaluation currently supports one device")
        if self._predictions:
            flat_metrics: dict[str, float] = {}
            for name, scorer in (("bleu", BLEU()), ("chrf", CHRF(word_order=2))):
                score = scorer.corpus_score(self._predictions, [self._references]).score
                self.log(f"{stage}/{name}", score, prog_bar=True)
                flat_metrics[f"{stage}/{name}"] = score
            direction_metrics: dict[str, dict[str, float]] = {}
            for direction, rows in self._evaluation_rows.items():
                predictions = [row["prediction"] for row in rows]
                references = [row["reference"] for row in rows]
                scores = {
                    "bleu": BLEU().corpus_score(predictions, [references]).score,
                    "chrf": CHRF(word_order=2)
                    .corpus_score(predictions, [references])
                    .score,
                }
                if all("comet" in row for row in rows):
                    scores["comet"] = sum(row["comet"] for row in rows) / len(rows)
                for name in SAC_METRICS:
                    if all(name in row for row in rows):
                        scores[name] = sum(row[name] for row in rows) / len(rows)
                direction_metrics[direction] = scores
                for name, score in scores.items():
                    self.log(f"{stage}/{direction}/{name}", score)
                    flat_metrics[f"{stage}/{direction}/{name}"] = score
            if direction_metrics:
                for name in set.intersection(
                    *(set(scores) for scores in direction_metrics.values())
                ):
                    value = sum(
                        scores[name] for scores in direction_metrics.values()
                    ) / len(direction_metrics)
                    self.log(f"{stage}/{name}_macro", value)
                    flat_metrics[f"{stage}/{name}_macro"] = value
                rows = [row for rows in self._evaluation_rows.values() for row in rows]
                flat_metrics[f"{stage}/reward"] = sum(
                    row["reward"] for row in rows
                ) / len(rows)
                report_dir = Path(self.trainer.default_root_dir) / "translation_metrics"
                report_dir.mkdir(parents=True, exist_ok=True)
                if self.hparams["save_predictions"]:
                    prediction_path = report_dir / f"{stage}_predictions.csv"
                    temporary = prediction_path.with_suffix(".csv.tmp")
                    with temporary.open(
                        "w", newline="", encoding="utf-8-sig"
                    ) as stream:
                        writer = csv.DictWriter(
                            stream,
                            fieldnames=[
                                "example_index",
                                "direction",
                                "source",
                                "reference",
                                "prediction",
                                "reward",
                                "comet",
                                *[
                                    name
                                    for name in SAC_METRICS
                                    if any(name in row for row in rows)
                                ],
                            ],
                        )
                        writer.writeheader()
                        writer.writerows(
                            sorted(rows, key=lambda row: row["example_index"])
                        )
                    temporary.replace(prediction_path)
                (report_dir / f"{stage}.json").write_text(
                    json.dumps(
                        {
                            "epoch": self.current_epoch,
                            "global_step": max(
                                self.global_step, self._loaded_optimizer_step
                            ),
                            "metrics": flat_metrics,
                            "directions": direction_metrics,
                            "counts": {
                                key: len(rows)
                                for key, rows in self._evaluation_rows.items()
                            },
                            "metric_scales": {
                                "bleu": "0-100",
                                "chrf": "0-100",
                                "comet": "raw model score",
                            },
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                if not self.trainer.sanity_checking:
                    if stage == "val":
                        self.kl_controller.observe_validation(flat_metrics)
                        if self._plateau is not None:
                            scheduler_metrics = self._plateau.step(
                                flat_metrics,
                                max(self.global_step, self._loaded_optimizer_step),
                            )
                            if scheduler_metrics:
                                append_record(
                                    report_dir / "lr_schedule.jsonl",
                                    {
                                        "global_step": self.global_step,
                                        **scheduler_metrics,
                                    },
                                )
                                for logger in self.trainer.loggers:
                                    logger.log_metrics(
                                        scheduler_metrics, step=self.global_step
                                    )
                    step = max(self.global_step, self._loaded_optimizer_step)
                    append_record(
                        report_dir / "evaluation_history.jsonl",
                        {
                            "stage": stage,
                            "global_step": step,
                            "epoch": self.current_epoch,
                            "metrics": flat_metrics,
                            "counts": {
                                key: len(value)
                                for key, value in self._evaluation_rows.items()
                            },
                        },
                    )
                    for logger in self.trainer.loggers:
                        logger.log_metrics(flat_metrics, step=step)
        self._predictions.clear()
        self._references.clear()
        self._evaluation_rows.clear()
        self.phase_timing.write()

    def on_validation_epoch_end(self) -> None:
        """Report corpus BLEU and chrF++ on the 0–100 scale."""
        self._evaluation_end("val")

    def on_test_epoch_end(self) -> None:
        """Report held-out corpus metrics."""
        self._evaluation_end("test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Construct the configured PyTorch optimizer using only trainable policy tensors."""
        parameters = [p for p in self.policy.parameters() if p.requires_grad]
        factory = self._optimizer_factory
        if factory is None:
            return torch.optim.AdamW(
                parameters,
                lr=self.hparams["learning_rate"],
                weight_decay=self.hparams["weight_decay"],
            )
        optimizer = factory(parameters)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a factory returning a PyTorch Optimizer")
        return optimizer

    def on_train_epoch_end(self) -> None:
        """Log parameter-drift controls separately from translation quality metrics."""
        optimizer = self.trainer.optimizers[0]
        if isinstance(optimizer, TransferAdamW):
            self.log_dict(
                {
                    f"train/transfer/{key}": value
                    for key, value in optimizer.drift_metrics().items()
                },
                on_step=False,
                on_epoch=True,
            )
        self.phase_timing.write()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Store adapter tensors and optimizer state; reconstruct the immutable frozen base."""
        checkpoint["telemetry_state"] = {"rollouts": self._rollout_counter}
        checkpoint["kl_controller"] = self.kl_controller.state_dict()
        if self._plateau is not None:
            checkpoint["validation_plateau"] = self._plateau.state_dict()
        if self.uses_lora:
            names = {name for name, p in self.named_parameters() if p.requires_grad}
            checkpoint["state_dict"] = {
                name: value
                for name, value in checkpoint["state_dict"].items()
                if name in names
            }
            checkpoint["adapter_checkpoint"] = {
                "model": self.hparams["model_name_or_path"],
                "base_revision": self.hparams["base_revision"],
            }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Validate the frozen-base identity before loading an adapter-only checkpoint."""
        self._rollout_counter = int(
            checkpoint.get("telemetry_state", {}).get("rollouts", 0)
        )
        self._loaded_optimizer_step = int(checkpoint.get("global_step", 0))
        self._restored_plateau = checkpoint.get("validation_plateau")
        cycle = checkpoint.get("cycle_state", {})
        self._reward_updates_seen = (
            int(cycle.get("stage_updates", 0))
            if self.hparams.get("cycle_stage") and cycle.get("stage") == "rl"
            else 0
            if self.hparams.get("cycle_stage")
            else self._loaded_optimizer_step
        )
        if "kl_controller" in checkpoint:
            self.kl_controller.load_state_dict(checkpoint["kl_controller"])
        identity = checkpoint.get("adapter_checkpoint")
        self._loading_adapter_checkpoint = bool(identity)
        if identity and (
            not self.uses_lora
            or identity["model"] != self.hparams["model_name_or_path"]
            or identity["base_revision"] != self.hparams["base_revision"]
        ):
            raise ValueError(
                "Adapter resume requires the same base model/revision and enabled LoRA"
            )

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load every trainable adapter tensor while retaining reconstructed quantization state."""
        if not self._loading_adapter_checkpoint:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        if set(state_dict) != expected:
            raise RuntimeError(
                "Adapter checkpoint has missing or unexpected trainable tensors"
            )
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        result.missing_keys.clear()  # Only frozen base tensors are intentionally omitted.
        self._loading_adapter_checkpoint = False
        return result

    def export_pretrained(self, path: str | Path) -> None:
        """Export policy/tokenizer, or PEFT adapter/tokenizer when LoRA is enabled."""
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
