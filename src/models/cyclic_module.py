"""Alternate objectives on one LoRA policy with a fixed, shared-base SFT reference."""

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.models.grpo_module import GRPOModule
from src.models.sft_module import TranslationSFTModule


class CyclicTranslationModule(TranslationSFTModule):
    """Keep the starting SFT adapter frozen; never re-anchor KL between cycles."""

    def __init__(
        self,
        reference_checkpoint: str,
        cycle_stage: str = "sft",
        stage_update_offset: int = 0,
        total_update_offset: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not self.uses_lora or cycle_stage not in {"sft", "rl"}:
            raise ValueError("Cyclic training requires LoRA and stage sft|rl")
        self.save_hyperparameters(
            {
                "reference_checkpoint": reference_checkpoint,
                "cycle_stage": cycle_stage,
                "stage_update_offset": stage_update_offset,
                "total_update_offset": total_update_offset,
            }
        )
        path = Path(reference_checkpoint)
        with path.open("rb") as stream:
            self.reference_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        anchor = torch.load(path, map_location="cpu", weights_only=False)
        identity = anchor.get("adapter_checkpoint", {})
        if identity != {
            "model": self.hparams["model_name_or_path"],
            "base_revision": self.hparams["base_revision"],
        }:
            raise ValueError(
                "Reference must use the same pinned foundation and LoRA layout"
            )
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        if set(anchor["state_dict"]) != expected:
            raise ValueError(
                "Reference adapter tensor layout does not match the policy"
            )
        self.policy.add_adapter(
            "cycle_reference", deepcopy(self.policy.peft_config["default"])
        )
        parameters = dict(self.named_parameters())
        with torch.no_grad():
            for name, value in anchor["state_dict"].items():
                target = parameters[name.replace(".default.", ".cycle_reference.")]
                if target.shape != value.shape or not bool(torch.isfinite(value).all()):
                    raise ValueError("Reference adapter contains invalid tensors")
                target.copy_(value)
        self.policy.set_adapter("default")
        self._freeze_reference()

    def _freeze_reference(self) -> None:
        for name, parameter in self.policy.named_parameters():
            if ".cycle_reference." in name:
                parameter.requires_grad_(False)

    @torch.no_grad()
    def _reference_logps(
        self, inputs: dict[str, Tensor], sequences: Tensor, mask: Tensor
    ) -> Tensor:
        self.policy.set_adapter("cycle_reference")
        try:
            return self._logps(self.policy, inputs, sequences, mask)
        finally:
            self.policy.set_adapter("default")
            self._freeze_reference()

    def training_step(self, batch: dict[str, list[str]], batch_idx: int) -> Any:
        if self.hparams["cycle_stage"] == "sft":
            return super().training_step(batch, batch_idx)
        return GRPOModule.training_step(self, batch, batch_idx)

    def _optimizer_step(self, optimizer: Any) -> None:
        # Stage-specific Adam moments and warmup continue across objective switches.
        fraction = (
            min(
                1.0,
                (self.hparams["stage_update_offset"] + self.global_step + 1)
                / self.warmup_steps,
            )
            if self.warmup_steps
            else 1.0
        )
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", self.hparams["learning_rate"])
            group["lr"] = group["initial_lr"] * fraction
        optimizer.step()

    def _evaluation_end(self, stage: str) -> None:
        self._loaded_optimizer_step = max(
            self._loaded_optimizer_step,
            self.hparams["total_update_offset"] + self.global_step,
        )
        super()._evaluation_end(stage)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["cycle_state"] = {
            "stage": self.hparams["cycle_stage"],
            "stage_updates": self.hparams["stage_update_offset"] + self.global_step,
            "total_updates": self.hparams["total_update_offset"] + self.global_step,
            "reference_sha256": self.reference_sha256,
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if "cycle_state" not in checkpoint:
            # Initial SFT is a weights/optimizer warm start, not an RL-controller resume.
            # The new fixed-reference ablation starts its own explicitly configured KL state.
            checkpoint = {
                key: value
                for key, value in checkpoint.items()
                if key != "kl_controller"
            }
        super().on_load_checkpoint(checkpoint)
        state = checkpoint.get("cycle_state")
        if state:
            if state["reference_sha256"] != self.reference_sha256:
                raise ValueError("The fixed SFT reference checkpoint changed")
            self._loaded_optimizer_step = state["total_updates"]
