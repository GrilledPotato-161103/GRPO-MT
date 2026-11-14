"""Supervised bilingual teacher forcing with the GRPO policy and optimizer interfaces."""

from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from src.data.components.multilingual_tokens import multilingual_ids
from src.models.grpo_module import GRPOModule


class TranslationSFTModule(GRPOModule):
    """Train on the SFT partition and evaluate both translation directions."""

    needs_reference = False

    def __init__(
        self,
        max_target_length: int = 128,
        microbatch_size: int | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if max_target_length < 2:
            raise ValueError("max_target_length must allow target text and EOS")
        if microbatch_size is not None and (
            isinstance(microbatch_size, bool)
            or not isinstance(microbatch_size, int)
            or microbatch_size < 1
        ):
            raise ValueError("SFT microbatch_size must be a positive integer")
        self.save_hyperparameters("max_target_length", "microbatch_size")

    def supervised_inputs(self, batch: dict[str, list[str]]) -> dict[str, Tensor]:
        """Mask causal prompts/padding while retaining target EOS, including EOS=PAD."""
        inputs = self._encode(
            batch["source"], batch.get("source_lang"), batch.get("target_lang")
        )
        limit = self.hparams["max_target_length"]
        if self.hparams["language_codes"]:
            source_langs = batch.get(
                "source_lang", [self.hparams["source_lang"]] * len(batch["source"])
            )
            target_langs = batch.get(
                "target_lang", [self.hparams["target_lang"]] * len(batch["source"])
            )
            ids = multilingual_ids(
                self.tokenizer,
                batch["reference"],
                source_langs,
                target_langs,
                self.hparams["language_codes"],
                targets=True,
            )
            if any(len(value) > limit for value in ids):
                if self.hparams["strict_prompt_length"]:
                    raise ValueError(
                        "Complete multilingual target plus language/EOS exceeds max_target_length"
                    )
                ids = multilingual_ids(
                    self.tokenizer,
                    batch["reference"],
                    source_langs,
                    target_langs,
                    self.hparams["language_codes"],
                    targets=True,
                    truncation=True,
                    max_length=limit,
                )
            native_labels = pad_sequence(
                [
                    torch.tensor(row, dtype=torch.long, device=self.device)
                    for row in ids
                ],
                batch_first=True,
                padding_value=-100,
            )
            # Shift BEFORE masking the deterministic language label; otherwise HF would
            # replace that decoder context token with PAD and train a different policy.
            decoder = torch.full_like(native_labels, self.tokenizer.pad_token_id)
            decoder[:, 0] = self.generation_config.decoder_start_token_id
            decoder[:, 1:] = native_labels[:, :-1]
            decoder.masked_fill_(decoder == -100, self.tokenizer.pad_token_id)
            native_labels[:, 0] = -100
            return {**inputs, "decoder_input_ids": decoder, "labels": native_labels}
        if self.is_encoder_decoder:
            targets = self.tokenizer(
                text_target=batch["reference"],
                padding=True,
                truncation=True,
                max_length=limit,
                return_tensors="pt",
            ).to(self.device)
            eos = self.tokenizer.eos_token_id
            if eos is None:
                raise ValueError("Supervised translation requires an EOS token")
            sequences = []
            for ids, mask in zip(
                targets["input_ids"], targets["attention_mask"], strict=True
            ):
                text_ids = ids[mask.bool()]
                if text_ids.numel() == 0 or int(text_ids[-1]) != eos:
                    text_ids = torch.cat(
                        [text_ids[: limit - 1], torch.tensor([eos], device=self.device)]
                    )
                sequences.append(text_ids)
            decoder_labels = pad_sequence(
                sequences, batch_first=True, padding_value=-100
            )
            return {**inputs, "labels": decoder_labels}
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Causal SFT requires an EOS token")
        target_texts = (
            batch["reference"]
            if self.hparams["use_chat_template"]
            else [" " + target for target in batch["reference"]]
        )
        if self.hparams["strict_prompt_length"]:
            complete = self.tokenizer(
                target_texts, add_special_tokens=False, truncation=False
            )["input_ids"]
            if any(len(ids) + 1 > limit for ids in complete):
                raise ValueError(
                    "Complete SFT target plus EOS exceeds max_target_length"
                )
        target_ids = self.tokenizer(
            target_texts,
            add_special_tokens=False,
            truncation=not self.hparams["strict_prompt_length"],
            max_length=limit - 1,
        )["input_ids"]
        sequences, labels, attention = [], [], []
        for index, ids in enumerate(target_ids):
            source = inputs["input_ids"][index][inputs["attention_mask"][index].bool()]
            target = torch.tensor([*ids, eos], dtype=torch.long, device=self.device)
            sequence = torch.cat([source, target])
            sequences.append(sequence)
            labels.append(torch.cat([torch.full_like(source, -100), target]))
            attention.append(torch.ones_like(sequence))
        return {
            "input_ids": pad_sequence(
                sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id
            ),
            "attention_mask": pad_sequence(
                attention, batch_first=True, padding_value=0
            ),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        }

    def training_step(self, batch: dict[str, list[str]], batch_idx: int) -> Tensor:
        with self.phase_timing.measure(
            "sft/batch_total", self.device, samples=len(batch["source"])
        ):
            return self._supervised_step(batch, batch_idx)

    def _supervised_step(self, batch: dict[str, list[str]], batch_idx: int) -> Tensor:
        """Apply one supervised optimizer update with foundation-preserving controls."""
        optimizer = cast(Any, self.optimizers())
        optimizer.zero_grad()
        self._prepare_gradient_scoring()
        inputs = self.supervised_inputs(batch)
        self._note_augmentation(batch)
        loss = self._backward_supervised(inputs)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.hparams["max_grad_norm"],
            gradient_clip_algorithm="norm",
        )
        with self.phase_timing.measure("sft/optimizer", self.device):
            self._optimizer_step(optimizer)
        self.log(
            "train/loss",
            loss.detach(),
            batch_size=len(batch["source"]),
            on_step=True,
            on_epoch=True,
        )
        return loss.detach()

    def _backward_supervised(
        self, inputs: dict[str, Tensor], backward_fn: Any = None
    ) -> Tensor:
        """Preserve the exact global token-mean CE with one graph per microbatch."""
        count = inputs["labels"].shape[0]
        size = self.hparams["microbatch_size"] or count
        tokens = (inputs["labels"] != -100).sum().clamp_min(1)
        total = torch.zeros((), device=self.device)
        backward = backward_fn or self.manual_backward
        for start in range(0, count, size):
            selected = {
                key: value[start : start + size] for key, value in inputs.items()
            }
            valid = (selected["labels"] != -100).sum()
            if not valid:
                continue
            with self.phase_timing.measure(
                "sft/current_forward",
                self.device,
                samples=len(selected["labels"]),
                input_tokens=int(selected["attention_mask"].sum()),
                output_tokens=int(valid),
            ):
                loss = self.policy(**selected, use_cache=False).loss * (valid / tokens)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite supervised loss")
            with self.phase_timing.measure(
                "sft/backward", self.device, samples=len(selected["labels"])
            ):
                backward(loss)
            total += loss.detach()
            del loss
        return total


def export_sft(checkpoint: Path, output: Path, load_dtype: str = "bfloat16") -> None:
    """Export a dense SFT base on CPU; merge adapters only after unquantized loading."""
    if output.exists():
        raise FileExistsError("SFT export exists; choose a new output directory")
    module = TranslationSFTModule.load_from_checkpoint(
        checkpoint,
        weights_only=False,
        map_location="cpu",
        quantization=None,
        load_dtype=load_dtype,
    )
    policy = (
        module.policy.merge_and_unload(safe_merge=True)
        if module.uses_lora
        else module.policy
    )
    output.mkdir(parents=True)
    policy.save_pretrained(output)
    module.tokenizer.save_pretrained(output)
