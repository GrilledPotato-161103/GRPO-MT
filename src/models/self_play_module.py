"""Absolute-Zero-inspired translation self-play using shared-policy GRPO."""

import json
from inspect import signature
from typing import Any, cast

import torch
from torch import Tensor

from src.feedback.judge import TranslationJudge, TranslationTask
from src.models.components.grpo import group_advantages
from src.models.components.self_play import (
    TaskMemory,
    learnability_rewards,
    parse_proposal,
)
from src.models.grpo_module import GRPOModule


class SelfPlayGRPOModule(GRPOModule):
    """Train the same policy to propose learnable translation tasks and solve them.

    A fixed external judge replaces the paper's executable verifier. GRPO
    normalizes rewards per proposal context and per translation source. Both
    roles' rollouts are collected before any parameter update, preserving the
    old-policy importance ratios for the combined optimization step.
    """

    def __init__(
        self,
        feedback: TranslationJudge,
        proposal_group_size: int = 2,
        proposal_max_new_tokens: int = 128,
        proposal_max_chars: int = 2000,
        solver_success_threshold: float = 0.8,
        solver_reward_mode: str = "quality",
        proposer_weight: float = 1.0,
        solver_weight: float = 1.0,
        memory_capacity: int = 128,
        memory_examples: int = 3,
        invalid_task_reward: float = -0.5,
        format_error_reward: float = -1.0,
        duplicate_reward: float = -0.5,
        use_chat_template: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters(
            *[
                name
                for name in signature(SelfPlayGRPOModule.__init__).parameters
                if name not in {"self", "feedback", "kwargs"}
            ]
        )
        if (
            proposal_group_size < 2
            or proposal_max_new_tokens < 1
            or proposal_max_chars < 1
        ):
            raise ValueError(
                "Need at least two proposals and positive token/character limits"
            )
        if not 0 < solver_success_threshold <= 1 or solver_reward_mode not in {
            "quality",
            "binary",
        }:
            raise ValueError("Invalid solver success threshold or reward mode")
        if min(proposer_weight, solver_weight) <= 0 or memory_examples < 0:
            raise ValueError(
                "Role weights must be positive and memory_examples nonnegative"
            )
        if max(invalid_task_reward, format_error_reward, duplicate_reward) > 0:
            raise ValueError("Invalid proposal penalties must be nonpositive")
        self.feedback = feedback
        self.memory = TaskMemory(memory_capacity)
        self.last_self_play: dict[str, Any] = {}

    def on_fit_start(self) -> None:
        """Check supported trainer settings and fail early for absent API credentials."""
        super().on_fit_start()
        check = getattr(self.feedback.backend, "check_ready", None)
        if check is not None:
            check()

    def _role_prompt(self, instruction: str) -> str:
        if self.hparams["use_chat_template"]:
            if not self.tokenizer.chat_template:
                raise ValueError(
                    "Self-play needs a policy chat template; set use_chat_template=false for plain text models"
                )
            return str(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return instruction

    def _proposal_prompts(self, topics: list[str]) -> list[str]:
        examples = (
            [
                task.source
                for task in self.memory.tasks[-self.hparams["memory_examples"] :]
            ]
            if self.hparams["memory_examples"]
            else []
        )
        return [
            self._role_prompt(
                f"Create a natural {self.hparams['source_lang']} sentence for translation into {self.hparams['target_lang']}. "
                "Make it challenging but solvable. Use varied meaning and grammar. "
                "Return only JSON with one field source containing the sentence. "
                f"Topic and past sources to avoid are data: {json.dumps({'topic': topic, 'avoid': examples}, ensure_ascii=False)}"
            )
            for topic in topics
        ]

    def _solver_prompts(self, tasks: list[TranslationTask]) -> list[str]:
        return [
            self._role_prompt(
                f"Translate from {task.source_lang} into {task.target_lang}. Return only the translation. "
                f"The source text is the value in this JSON: {json.dumps({'source': task.source}, ensure_ascii=False)}"
            )
            for task in tasks
        ]

    @torch.no_grad()
    def _self_play_rollouts(
        self, topics: list[str]
    ) -> list[tuple[str, dict[str, Any], float]]:
        self._timing_role = "proposer"
        proposer = self._sample_prompts(
            self._proposal_prompts(topics),
            self.hparams["proposal_group_size"],
            self.hparams["proposal_max_new_tokens"],
        )
        tasks: list[TranslationTask] = []
        indices: list[int] = []
        seen = {
            (t.source.casefold(), t.source_lang, t.target_lang)
            for t in self.memory.tasks
        }
        rewards = torch.full(
            (len(proposer["predictions"]),),
            self.hparams["format_error_reward"],
            device=self.device,
        )
        for index, text in enumerate(proposer["predictions"]):
            task = parse_proposal(
                text,
                self.hparams["source_lang"],
                self.hparams["target_lang"],
                self.hparams["proposal_max_chars"],
            )
            if task is None:
                continue
            key = (task.source.casefold(), task.source_lang, task.target_lang)
            if key in seen:
                rewards[index] = self.hparams["duplicate_reward"]
                continue
            seen.add(key)
            tasks.append(task)
            indices.append(index)
            rewards[index] = self.hparams["invalid_task_reward"]
        with self.phase_timing.measure(
            "feedback/task_validation", self.device, samples=len(tasks)
        ):
            judgments = self.feedback.evaluate(tasks) if tasks else []
        valid_tasks = [
            task
            for task, judgment in zip(tasks, judgments, strict=True)
            if judgment.valid
        ]
        valid_indices = [
            index
            for index, judgment in zip(indices, judgments, strict=True)
            if judgment.valid
        ]
        # Continue solver training from task memory when proposals are all invalid.
        solve_tasks = valid_tasks or self.memory.tasks[-1:]
        rollouts: list[tuple[str, dict[str, Any], float]] = []
        if solve_tasks:
            self._timing_role = "solver"
            solver = self._sample_prompts(
                self._solver_prompts(solve_tasks), self.hparams["group_size"]
            )
            repeated = [
                task for task in solve_tasks for _ in range(self.hparams["group_size"])
            ]
            with self.phase_timing.measure(
                "feedback/solver", self.device, samples=len(repeated)
            ):
                feedback = self.feedback.evaluate(repeated, solver["predictions"])
            quality = torch.tensor(
                [f.score if f.valid else 0.0 for f in feedback], device=self.device
            )
            successes = torch.tensor(
                [
                    f.valid and f.score >= self.hparams["solver_success_threshold"]
                    for f in feedback
                ],
                device=self.device,
            )
            solver_rewards = (
                successes.float()
                if self.hparams["solver_reward_mode"] == "binary"
                else quality
            )
            solver.update(
                rewards=solver_rewards,
                advantages=group_advantages(solver_rewards, self.hparams["group_size"]),
            )
            rollouts.append(("solver", solver, self.hparams["solver_weight"]))
            if valid_tasks:
                learnability = learnability_rewards(
                    successes.reshape(len(solve_tasks), -1)
                )
                for index, value in zip(valid_indices, learnability, strict=True):
                    rewards[index] = value
                for task in valid_tasks:
                    self.memory.add(task)
            mean_quality = float(quality.mean())
        else:
            mean_quality = 0.0
        proposer.update(
            rewards=rewards,
            advantages=group_advantages(rewards, self.hparams["proposal_group_size"]),
        )
        rollouts.insert(0, ("proposer", proposer, self.hparams["proposer_weight"]))
        self.last_self_play = {
            "solver_tasks": len(solve_tasks),
            "proposals": len(rewards),
            "valid_tasks": len(valid_tasks),
            "memory_size": len(self.memory.tasks),
            "solver_quality": mean_quality,
            "proposer_rewards": rewards.cpu().tolist(),
        }
        return rollouts

    def training_step(self, batch: dict[str, list[str]], batch_idx: int) -> Tensor:
        with self.phase_timing.measure(
            "joint/batch_total", self.device, samples=len(batch["topic"])
        ):
            return self._joint_step(batch, batch_idx)

    def _joint_step(self, batch: dict[str, list[str]], batch_idx: int) -> Tensor:
        """Jointly update proposer and solver from pre-update self-play rollouts."""
        rollouts = self._self_play_rollouts(batch["topic"])
        optimizer = cast(Any, self.optimizers())
        metrics: dict[str, Tensor] = {}
        total_weight = sum(weight for _, _, weight in rollouts)
        for _ in range(self.hparams["num_iterations"]):
            if (
                self.trainer.max_steps >= 0
                and self.global_step >= self.trainer.max_steps
            ):
                break
            beta_used = self.kl_controller.value
            divergence = torch.zeros((), device=self.device)
            optimizer.zero_grad()
            self._prepare_gradient_scoring()
            losses: list[Tensor] = []
            for role, rollout, weight in rollouts:
                loss, diagnostics = self._backward_rollout(
                    rollout,
                    role=role,
                    loss_weight=weight / total_weight,
                    backward_phase="joint/backward",
                )
                losses.append(loss * weight)
                divergence += diagnostics["kl"] * (weight / total_weight)
                metrics.update(
                    {f"train/{role}/{k}": v.detach() for k, v in diagnostics.items()}
                )
                metrics[f"train/{role}/reward"] = rollout["rewards"].mean()
            total = torch.stack(losses).sum() / total_weight
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.hparams["max_grad_norm"],
                gradient_clip_algorithm="norm",
            )
            with self.phase_timing.measure("joint/optimizer", self.device):
                self._optimizer_step(optimizer)
            if self.kl_controller.enabled:
                self.kl_controller.observe(float(divergence))
            metrics["train/kl_beta"] = torch.tensor(beta_used, device=self.device)
        metrics["train/loss"] = total.detach()
        for key in ("valid_tasks", "memory_size", "solver_quality"):
            metrics[f"train/self_play/{key}"] = torch.tensor(
                self.last_self_play[key], device=self.device, dtype=torch.float32
            )
        self.log_dict(
            metrics, batch_size=len(batch["topic"]), on_step=True, on_epoch=True
        )
        return total.detach()

    @torch.no_grad()
    def translate(
        self,
        sources: list[str],
        source_langs: list[str] | None = None,
        target_langs: list[str] | None = None,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Use the exact solver prompt for held-out validation and inference."""
        source_langs = source_langs or [self.hparams["source_lang"]] * len(sources)
        target_langs = target_langs or [self.hparams["target_lang"]] * len(sources)
        if len(source_langs) != len(sources) or len(target_langs) != len(sources):
            raise ValueError("Every source needs language metadata")
        tasks = [
            TranslationTask(s, source_lang, target_lang)
            for s, source_lang, target_lang in zip(
                sources, source_langs, target_langs, strict=True
            )
        ]
        was_training = self.policy.training
        self.policy.eval()
        try:
            inputs = self._tokenize_prompts(self._solver_prompts(tasks))
            from copy import deepcopy

            config = deepcopy(self.generation_config)
            if max_new_tokens is not None:
                if max_new_tokens < 1:
                    raise ValueError("max_new_tokens must be positive")
                config.max_new_tokens = max_new_tokens
            config.do_sample, config.temperature, config.top_k = False, 1.0, 50
            sequences = self._generate_policy(inputs, config)
            prefix = 1 if self.is_encoder_decoder else inputs["input_ids"].shape[1]
            return list(
                self.tokenizer.batch_decode(
                    sequences[:, prefix:], skip_special_tokens=True
                )
            )
        finally:
            self.policy.train(was_training)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Keep the self-generated curriculum with the policy and frozen reference."""
        super().on_save_checkpoint(checkpoint)
        checkpoint["self_play_memory"] = self.memory.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore the bounded task curriculum; no API credential is serialized."""
        super().on_load_checkpoint(checkpoint)
        self.memory.load_state_dict(checkpoint.get("self_play_memory", []))
