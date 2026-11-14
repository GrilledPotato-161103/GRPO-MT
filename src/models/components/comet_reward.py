"""Lazy frozen COMETKiwi inference without constructing a nested Lightning Trainer."""

import math
import time
from pathlib import Path
from typing import Any

import torch

from src.utils.credentials import environment_secret


class CometScorer:
    """Score source/hypothesis pairs using the official COMETKiwi model.

    CPU is the default so the policy's GPU budget is unaffected. The scorer is
    a plain Python object: its frozen model is excluded from policy checkpoints
    and optimizers. Optional injection supports lightweight interface tests.
    """

    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-cometkiwi-da",
        device: str = "cpu",
        batch_size: int = 8,
        cache_dir: str | None = None,
        model: Any = None,
        api_key_env: str = "HUGGING_FACE_API_KEY",
        reference_based: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("COMET batch_size must be positive")
        parsed_device = torch.device(device)
        if parsed_device.type not in {"cpu", "cuda"}:
            raise ValueError("COMET device must be cpu or cuda[:index]")
        if parsed_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable for COMET")
        self.model_name, self.device = model_name, parsed_device
        self.batch_size, self.cache_dir = batch_size, cache_dir
        self.model = model
        self.api_key_env, self.reference_based = api_key_env, reference_based
        self._ready = False
        self.last_timings: dict[str, float] = {}
        self.last_input_tokens = 0

    def _load(self) -> Any:
        if self._ready:
            return self.model
        if self.model is None:
            try:
                from comet import load_from_checkpoint
                from huggingface_hub import snapshot_download
            except ImportError as error:
                raise ImportError(
                    "Install the COMET extra: python -m pip install -e '.[comet]'"
                ) from error
            if self.cache_dir:
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            if Path(self.model_name).is_file():
                checkpoint = self.model_name
            else:
                token = environment_secret(self.api_key_env)
                if not token:
                    raise ValueError(
                        f"Set {self.api_key_env} and accept the model's Hugging Face conditions"
                    )
                try:
                    snapshot = snapshot_download(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        token=token,
                        allow_patterns=["checkpoints/*.ckpt", "hparams.yaml"],
                    )
                except (OSError, ValueError):
                    raise RuntimeError(
                        f"Cannot download COMET model; verify {self.api_key_env} access and accepted conditions"
                    ) from None
                checkpoint = str(Path(snapshot) / "checkpoints/model.ckpt")
            self.model = load_from_checkpoint(checkpoint)
        self.model.to(self.device).eval().requires_grad_(False)
        self._ready = True
        return self.model

    @torch.inference_mode()
    def __call__(
        self,
        sources: list[str],
        predictions: list[str],
        references: list[str] | None = None,
    ) -> list[float]:
        """Return raw segment scores in input order; reject malformed outputs."""
        if len(sources) != len(predictions) or (
            references is not None and len(references) != len(predictions)
        ):
            raise ValueError(
                "COMET needs aligned sources, predictions, and optional references"
            )
        if self.reference_based and references is None:
            raise ValueError("Reference-based COMET needs references")
        if not predictions:
            return []
        if any(
            not isinstance(text, str)
            for text in (*sources, *predictions, *(references or []))
        ):
            raise ValueError("COMET inputs must be text")
        started = time.perf_counter()
        model = self._load()
        self.last_timings = {
            "load": time.perf_counter() - started,
            "tokenization": 0.0,
            "forward": 0.0,
        }
        self.last_input_tokens = 0
        scores: list[float] = []
        rows = [
            {"src": source, "mt": prediction}
            for source, prediction in zip(sources, predictions, strict=True)
        ]
        if self.reference_based:
            assert references is not None
            for row, reference in zip(rows, references, strict=True):
                row["ref"] = reference
        for offset in range(0, len(rows), self.batch_size):
            started = time.perf_counter()
            batch = model.prepare_sample(
                rows[offset : offset + self.batch_size], stage="predict"
            )
            batches = [batch] if isinstance(batch, dict) else list(batch)
            inputs = [
                {
                    key: value.to(self.device)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in item.items()
                }
                for item in batches
            ]
            self.last_input_tokens += sum(
                int(tensor.sum())
                for item in inputs
                for key, tensor in item.items()
                if key.endswith("attention_mask")
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.last_timings["tokenization"] += time.perf_counter() - started
            started = time.perf_counter()
            # Unified COMETKiwi returns a tuple of joint-encoded inputs. The
            # official predict_step averages its sequence scores when there are
            # multiple inputs; regression COMET returns a single dictionary.
            # Keep the fixed evaluator independent of the policy's AMP context.
            with torch.autocast(device_type=self.device.type, enabled=False):
                output = (
                    torch.stack([model(**item).score for item in inputs])
                    .mean(0)
                    .detach()
                    .float()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                )
            self.last_timings["forward"] += time.perf_counter() - started
            if len(output) != len(rows[offset : offset + self.batch_size]) or any(
                not math.isfinite(value) for value in output
            ):
                raise ValueError("COMET must return one finite scalar per translation")
            scores.extend(float(value) for value in output)
        return scores
