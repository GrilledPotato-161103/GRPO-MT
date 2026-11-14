"""Translation text loading independent of tokenization and model architecture."""

import csv
import hashlib
import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, get_worker_info

from src.data.components.augmentation import TranslationAugmentation, make_augmentation


class TranslationDataset(Dataset[dict[str, str]]):
    """Store normalized source/reference pairs."""

    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        bidirectional: bool = False,
        source_lang: str = "English",
        target_lang: str = "Vietnamese",
        source_code: str = "en",
        target_code: str = "vi",
        augmentation: TranslationAugmentation | None = None,
        seed: int = 42,
        paired_batches: bool = False,
    ) -> None:
        self.rows = rows
        if paired_batches and not bidirectional:
            raise ValueError("Paired translation data requires bidirectional views")
        self.bidirectional = bidirectional
        self.source_lang, self.target_lang = source_lang, target_lang
        self.source_code, self.target_code = source_code, target_code
        self.augmentation, self.seed = augmentation, seed
        self.paired_batches = paired_batches
        self._paired_cache: tuple[int, int, dict[str, str]] | None = None
        self._rng: random.Random | None = None
        self._worker_seed: int | None = None

    def __len__(self) -> int:
        return len(self.rows) * (2 if self.bidirectional else 1)

    def unaugmented_item(self, index: int) -> dict[str, str]:
        """Read a directional view without advancing the online augmentation RNG."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        row_index, reverse = (
            (index // 2, bool(index % 2)) if self.bidirectional else (index, False)
        )
        row = dict(self.rows[row_index])
        if reverse:
            row["source"], row["reference"] = row["reference"], row["source"]
        if self.bidirectional:
            row.update(
                source_lang=self.target_lang if reverse else self.source_lang,
                target_lang=self.source_lang if reverse else self.target_lang,
                direction=f"{self.target_code}_{self.source_code}"
                if reverse
                else f"{self.source_code}_{self.target_code}",
            )
        if self.paired_batches:
            row["pair_id"] = str(row_index)
        return row

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.unaugmented_item(index)
        row_index, reverse = (
            (index // 2, bool(index % 2)) if self.bidirectional else (index, False)
        )
        if self.augmentation is not None:
            worker = get_worker_info()
            worker_seed = worker.seed if worker is not None else self.seed
            if self._rng is None or self._worker_seed != worker_seed:
                self._rng, self._worker_seed = random.Random(worker_seed), worker_seed
                self._paired_cache = None
            if self.paired_batches:
                return self._paired_augmented_item(row, row_index, reverse)
            partner = None
            if len(self.rows) > 1:
                partner_index = self._rng.randrange(len(self.rows) - 1)
                partner_index += partner_index >= row_index
                pair = self.rows[partner_index]
                partner = (
                    (pair["reference"], pair["source"])
                    if reverse
                    else (pair["source"], pair["reference"])
                )
            row.update(
                self.augmentation.apply(
                    row["source"], row["reference"], partner, self._rng
                )
            )
        return row

    def _paired_augmented_item(
        self, row: dict[str, str], index: int, reverse: bool
    ) -> dict[str, str]:
        """Concatenate once per pair, then perturb each input's formatting independently."""
        assert self._rng is not None and self.augmentation is not None
        bit = 1 << int(reverse)
        cached = self._paired_cache
        if cached is None or cached[0] != index or cached[1] & bit:
            clean = self.rows[index]
            partner = None
            if len(self.rows) > 1:
                other = self._rng.randrange(len(self.rows) - 1)
                other += other >= index
                partner = (self.rows[other]["source"], self.rows[other]["reference"])
            augmented = self.augmentation.apply(
                clean["source"], clean["reference"], partner, self._rng
            )
            seen = 0
        else:
            _, seen, augmented = cached
        self._paired_cache = (
            None if (seen | bit) == 3 else (index, seen | bit, augmented)
        )
        if reverse:
            result = self.augmentation.apply(
                augmented["reference"], augmented["clean_source"], None, self._rng
            )
            if "concat" in augmented["augmentation"]:
                operations = [
                    name for name in result["augmentation"].split("+") if name != "none"
                ]
                result["augmentation"] = "+".join(["concat", *operations])
            row.update(result)
        else:
            row.update(augmented)
        return row


def collate_translation(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """Keep raw text so generation can tokenize for either architecture."""
    if not rows or any(row.keys() != rows[0].keys() for row in rows):
        raise ValueError("Translation batches require rows with matching fields")
    return {key: [row[key] for row in rows] for key in rows[0]}


class TranslationDataModule(L.LightningDataModule):
    """Read CSV/JSONL or Hugging Face datasets with flat or translation columns.

    A missing validation file is split from training deterministically. Test data
    is never used for validation. Explicit splits are not silently substituted.
    """

    def __init__(
        self,
        train_file: str | None = None,
        val_file: str | None = None,
        test_file: str | None = None,
        source_column: str = "en",
        target_column: str = "vi",
        translation_column: str | None = None,
        dataset_name: str | None = None,
        dataset_config: str | None = None,
        train_split: str = "train",
        val_split: str | None = None,
        test_split: str | None = None,
        val_fraction: float = 0.05,
        batch_size: int = 2,
        evaluation_batch_size: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        bidirectional: bool = False,
        source_lang: str = "English",
        target_lang: str = "Vietnamese",
        source_code: str = "en",
        target_code: str = "vi",
        augmentation: dict[str, Any] | None = None,
        resume_skip_examples: int = 0,
        token_budget: int | None = None,
        budget_tokenizer_name_or_path: str | None = None,
        budget_tokenizer_revision: str | None = None,
        token_budget_source_cap: int = 512,
        token_budget_target_cap: int = 128,
        token_budget_prompt_template: str = "Translate {source_lang} to {target_lang}. Return only the translation.\nSource: {source}\nTranslation:",
        length_batching: dict[str, Any] | None = None,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        disk_backed: bool = False,
        paired_batches: bool = False,
        resume_batch_plan: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if evaluation_batch_size is not None and (
            isinstance(evaluation_batch_size, bool)
            or not isinstance(evaluation_batch_size, int)
            or evaluation_batch_size < 1
            or (paired_batches and evaluation_batch_size % 2)
        ):
            raise ValueError(
                "evaluation_batch_size must be positive and even for paired batches"
            )
        if paired_batches and (
            not bidirectional
            or batch_size % 2
            or batch_size < 2
            or token_budget is not None
            or resume_skip_examples
        ):
            raise ValueError(
                "Paired batches require bidirectional views, an even batch size, and no legacy token-budget/rebatch offsets"
            )
        if (
            isinstance(prefetch_factor, bool)
            or not isinstance(prefetch_factor, int)
            or prefetch_factor < 1
        ):
            raise ValueError("prefetch_factor must be a positive integer")
        if token_budget is not None and (
            token_budget < 1 or not budget_tokenizer_name_or_path or num_workers != 0
        ):
            raise ValueError(
                "Token budgeting requires a tokenizer, positive budget, and zero workers"
            )
        if resume_skip_examples < 0 or (resume_skip_examples and num_workers != 0):
            raise ValueError(
                "Rebatch resume requires a nonnegative offset and zero workers"
            )
        if batch_size < 1 or num_workers < 0 or not 0 < val_fraction < 1:
            raise ValueError("Invalid batch size, worker count, or validation fraction")
        if dataset_name is None and train_file is None:
            raise ValueError("Provide train_file or dataset_name")
        self.splits: dict[str, TranslationDataset] = {}
        self.augmentation = make_augmentation(augmentation)
        if isinstance(length_batching, DictConfig):
            length_batching = cast(
                dict[str, Any], OmegaConf.to_container(length_batching, resolve=True)
            )
        self.length_batching = {
            "enabled": False,
            "bucket_width": 16,
            "max_tokens": None,
            "candidate_multiplier": 1,
            "reserve_target_tokens": 0,
            **(length_batching or {}),
        }
        if self.length_batching["enabled"]:
            if token_budget is not None or resume_skip_examples:
                raise ValueError(
                    "Length batching cannot be combined with legacy token-budget/rebatch offsets"
                )
            if not self.length_batching.get("tokenizer_name_or_path"):
                raise ValueError("Length batching requires tokenizer_name_or_path")
            for name in (
                "source_cap",
                "target_cap",
                "bucket_width",
                "candidate_multiplier",
            ):
                value = self.length_batching.get(name, 0)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"length_batching.{name} must be a positive integer"
                    )
            for name in ("max_tokens", "reserve_target_tokens"):
                value = self.length_batching[name]
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < (1 if name == "max_tokens" else 0)
                ):
                    raise ValueError(f"Invalid length_batching.{name}")

    def _worker_options(self) -> dict[str, Any]:
        if not self.hparams["num_workers"]:
            return {}
        return {
            "persistent_workers": self.hparams["persistent_workers"],
            "prefetch_factor": self.hparams["prefetch_factor"],
        }

    def _length_loader(self) -> DataLoader[Any]:
        from transformers import AutoTokenizer

        from src.data.components.length_batching import (
            LengthBucketBatchSampler,
            PairedBatchSampler,
            TranslationCollator,
            translation_lengths,
        )

        options = self.length_batching
        tokenizer = cast(Any, AutoTokenizer).from_pretrained(
            options["tokenizer_name_or_path"],
            revision=options.get("tokenizer_revision"),
        )
        length_options = {
            "source_cap": options["source_cap"],
            "target_cap": options["target_cap"],
            "prompt_template": options.get(
                "prompt_template", self.hparams["token_budget_prompt_template"]
            ),
            "source_lang": self.hparams["source_lang"],
            "target_lang": self.hparams["target_lang"],
            "use_chat_template": options.get("use_chat_template", False),
            "chat_template_kwargs": options.get("chat_template_kwargs"),
            "language_codes": options.get("language_codes"),
        }
        dataset = self.splits["train"]
        lengths: list[tuple[int, int]] = []
        cache = None
        if options.get("cache_dir") and self.hparams["train_file"]:
            path = Path(self.hparams["train_file"])
            stat = path.stat()
            identity = {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "bidirectional": self.hparams["bidirectional"],
                "columns": [
                    self.hparams[k]
                    for k in ("source_column", "target_column", "translation_column")
                ],
                "tokenizer": options["tokenizer_name_or_path"],
                "revision": options.get("tokenizer_revision"),
                "length_options": length_options,
                "tokenizer_state": hashlib.sha256(
                    tokenizer.backend_tokenizer.to_str().encode()
                ).hexdigest()
                if getattr(tokenizer, "backend_tokenizer", None) is not None
                else hashlib.sha256(
                    json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
                ).hexdigest(),
            }
            key = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()
            cache = Path(options["cache_dir"]) / f"{key}.npy"
            if cache.exists():
                import numpy as np

                stored = np.load(cache, mmap_mode="r", allow_pickle=False)
                if stored.shape != (len(dataset), 2) or bool((stored < 1).any()):
                    raise ValueError(
                        "Invalid token-length cache; remove the corrupt cache file"
                    )
                lengths = [(int(a), int(b)) for a, b in stored]
        if not lengths:
            # Parallelize the one-time parent index, then disable Rayon before
            # worker processes fork and tokenize their own collated batches.
            os.environ.setdefault(
                "RAYON_NUM_THREADS", str(max(1, self.hparams["num_workers"]))
            )
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            try:
                for start in range(0, len(dataset), 512):
                    rows = [
                        dataset.unaugmented_item(i)
                        for i in range(start, min(start + 512, len(dataset)))
                    ]
                    lengths.extend(
                        translation_lengths(rows, tokenizer, **length_options)
                    )
                    if start % 65536 == 0:
                        print(
                            f"Token-length index: {len(lengths):,}/{len(dataset):,}",
                            flush=True,
                        )
            finally:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if cache is not None:
                import numpy as np

                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_name(cache.name + f".{os.getpid()}.tmp")
                with temporary.open("wb") as stream:
                    np.save(
                        stream, np.asarray(lengths, dtype=np.int32), allow_pickle=False
                    )
                temporary.replace(cache)
        paired = self.hparams["paired_batches"]
        packed_lengths = (
            [
                (
                    max(lengths[i][0], lengths[i + 1][0]),
                    max(lengths[i][1], lengths[i + 1][1]),
                )
                for i in range(0, len(lengths), 2)
            ]
            if paired
            else lengths
        )
        length_sampler = LengthBucketBatchSampler(
            packed_lengths,
            self.hparams["batch_size"] // 2 if paired else self.hparams["batch_size"],
            bucket_width=options["bucket_width"],
            seed=self.hparams["seed"],
            token_budget=options["max_tokens"],
            candidate_multiplier=options["candidate_multiplier"] * (2 if paired else 1),
            reserve_target_tokens=options["reserve_target_tokens"],
            # Online concatenation can grow a clean source up to the cap.
            reserve_source_tokens=options["source_cap"]
            if self.augmentation is not None
            else 0,
        )
        batch_sampler: Any = (
            PairedBatchSampler(
                len(dataset.rows),
                self.hparams["batch_size"],
                pair_sampler=length_sampler,
            )
            if paired
            else length_sampler
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            collate_fn=TranslationCollator(tokenizer, **length_options),
            generator=torch.Generator().manual_seed(self.hparams["seed"]),
            **self._worker_options(),
        )

    def _dataset(
        self, rows: Sequence[dict[str, str]], training: bool = False
    ) -> TranslationDataset:
        return TranslationDataset(
            rows,
            bidirectional=self.hparams["bidirectional"],
            source_lang=self.hparams["source_lang"],
            target_lang=self.hparams["target_lang"],
            source_code=self.hparams["source_code"],
            target_code=self.hparams["target_code"],
            augmentation=self.augmentation if training else None,
            seed=self.hparams["seed"],
            paired_batches=self.hparams["paired_batches"],
        )

    def _read(self, path: str | None, split: str | None) -> Sequence[dict[str, str]]:
        rows: Any
        if self.hparams["dataset_name"]:
            if split is None:
                raise ValueError(
                    "Provide an explicit Hugging Face split for the requested stage"
                )
            from datasets import load_dataset

            rows = load_dataset(
                self.hparams["dataset_name"],
                self.hparams["dataset_config"],
                split=split,
            )
        elif path:
            if self.hparams["disk_backed"]:
                from src.data.components.indexed_translation import (
                    IndexedTranslationRows,
                )

                return IndexedTranslationRows(
                    path,
                    self.hparams["source_column"],
                    self.hparams["target_column"],
                    self.hparams["translation_column"],
                )
            file = Path(path)
            with file.open(encoding="utf-8-sig", newline="") as stream:
                if file.suffix == ".csv":
                    rows = list(csv.DictReader(stream))
                elif file.suffix in {".jsonl", ".ndjson"}:
                    rows = [json.loads(line) for line in stream if line.strip()]
                else:
                    raise ValueError("Local data must be CSV or JSONL")
        else:
            raise ValueError("No data configured for requested split")
        normalized = []
        for index, row in enumerate(rows):
            pair = (
                row[self.hparams["translation_column"]]
                if self.hparams["translation_column"]
                else row
            )
            source, target = (
                pair[self.hparams["source_column"]],
                pair[self.hparams["target_column"]],
            )
            if (
                not isinstance(source, str)
                or not isinstance(target, str)
                or not source.strip()
                or not target.strip()
            ):
                raise ValueError(f"Empty or non-text translation pair at row {index}")
            normalized.append({"source": source, "reference": target})
        if not normalized:
            raise ValueError("Translation split is empty")
        return normalized

    def setup(self, stage: str | None = None) -> None:
        """Load only the splits needed by the Lightning stage."""
        if stage in (None, "fit", "validate") and "val" not in self.splits:
            train = self._read(self.hparams["train_file"], self.hparams["train_split"])
            if self.hparams["val_file"] or self.hparams["val_split"]:
                val = self._read(self.hparams["val_file"], self.hparams["val_split"])
            else:
                if self.hparams["disk_backed"]:
                    raise ValueError(
                        "Disk-backed training requires an explicit validation split"
                    )
                if len(train) < 2:
                    raise ValueError(
                        "Need at least two training pairs for a validation split"
                    )
                order = torch.randperm(
                    len(train),
                    generator=torch.Generator().manual_seed(self.hparams["seed"]),
                ).tolist()
                n_val = max(
                    1,
                    min(
                        len(train) - 1, round(len(train) * self.hparams["val_fraction"])
                    ),
                )
                val = [train[i] for i in order[:n_val]]
                train = [train[i] for i in order[n_val:]]
            self.splits.update(
                train=self._dataset(train, training=True), val=self._dataset(val)
            )
        if stage in (None, "test") and "test" not in self.splits:
            self.splits["test"] = self._dataset(
                self._read(self.hparams["test_file"], self.hparams["test_split"])
            )

    def _loader(self, split: str, shuffle: bool = False) -> DataLoader[Any]:
        batch_size = (
            self.hparams["batch_size"]
            if split == "train"
            else (self.hparams["evaluation_batch_size"] or self.hparams["batch_size"])
        )
        if (
            split == "train"
            and self.hparams["resume_batch_plan"]
            and (self.trainer is None or self.trainer.current_epoch == 0)
        ):
            # An explicit plan preserves the remaining examples when the batch
            # size changes after a checkpoint. Lightning loop counters must be
            # reset separately; weights and optimizer state remain untouched.
            plan = json.loads(Path(self.hparams["resume_batch_plan"]).read_text())
            batches = plan["batches"]
            if not batches or any(
                not batch
                or len(batch) > self.hparams["batch_size"]
                or any(
                    type(i) is not int or not 0 <= i < len(self.splits[split])
                    for i in batch
                )
                for batch in batches
            ):
                raise ValueError("Invalid resumed batch plan")
            if self.hparams["paired_batches"] and any(
                len(batch) % 2
                or any(a % 2 or b != a + 1 for a, b in zip(batch[::2], batch[1::2]))
                for batch in batches
            ):
                raise ValueError("Resumed batch plan must preserve bilingual pairs")
            return DataLoader(
                self.splits[split],
                batch_sampler=batches,
                num_workers=self.hparams["num_workers"],
                pin_memory=self.hparams["pin_memory"],
                collate_fn=collate_translation,
                generator=torch.Generator().manual_seed(self.hparams["seed"]),
                **self._worker_options(),
            )
        if split == "train" and self.length_batching["enabled"]:
            return self._length_loader()
        if self.hparams["paired_batches"]:
            from src.data.components.length_batching import PairedBatchSampler

            return DataLoader(
                self.splits[split],
                batch_sampler=PairedBatchSampler(
                    len(self.splits[split].rows),
                    batch_size,
                    self.hparams["seed"],
                    shuffle,
                ),
                num_workers=self.hparams["num_workers"],
                pin_memory=self.hparams["pin_memory"],
                collate_fn=collate_translation,
                generator=torch.Generator().manual_seed(self.hparams["seed"]),
                **self._worker_options(),
            )
        sampler = None
        if split == "train" and (
            self.hparams["resume_skip_examples"] or self.hparams["token_budget"]
        ):
            generator = torch.Generator().manual_seed(self.hparams["seed"])
            # Match DataLoader's worker-base-seed draw before RandomSampler.
            torch.empty((), dtype=torch.int64).random_(generator=generator)
            order = torch.randperm(
                len(self.splits[split]), generator=generator
            ).tolist()
            offset = (
                self.hparams["resume_skip_examples"]
                if self.trainer is None or self.trainer.current_epoch == 0
                else 0
            )
            if offset >= len(order):
                raise ValueError("Rebatch offset leaves no training examples")
            for index in order[:offset]:
                self.splits[split][
                    index
                ]  # Restore the single-worker augmentation stream.
            sampler, shuffle = order[offset:], False
            if self.hparams["token_budget"]:
                from transformers import AutoTokenizer

                from src.data.components.token_budget import prepare_token_batches

                tokenizer = cast(Any, AutoTokenizer).from_pretrained(
                    self.hparams["budget_tokenizer_name_or_path"],
                    revision=self.hparams["budget_tokenizer_revision"],
                )
                rows = [self.splits[split][index] for index in sampler]
                prepared, batches = prepare_token_batches(
                    rows,
                    tokenizer,
                    self.hparams["token_budget"],
                    self.hparams["token_budget_source_cap"],
                    self.hparams["token_budget_target_cap"],
                    self.hparams["token_budget_prompt_template"],
                )
                return DataLoader(
                    prepared,
                    batch_sampler=batches,
                    num_workers=0,
                    pin_memory=self.hparams["pin_memory"],
                    collate_fn=collate_translation,
                )
        return DataLoader(
            self.splits[split],
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            collate_fn=collate_translation,
            generator=torch.Generator().manual_seed(self.hparams["seed"]),
            **self._worker_options(),
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return shuffled training pairs."""
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        """Return validation pairs."""
        return self._loader("val")

    def test_dataloader(self) -> DataLoader[Any]:
        """Return held-out test pairs."""
        return self._loader("test")
