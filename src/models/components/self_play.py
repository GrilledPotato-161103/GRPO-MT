"""Task parsing, bounded task memory, and learnability rewards for translation."""

import json

import torch
from torch import Tensor

from src.feedback.judge import TranslationTask


def parse_proposal(
    text: str, source_lang: str, target_lang: str, max_chars: int = 2000
) -> TranslationTask | None:
    """Require a complete JSON source proposal; never execute model-generated code."""
    try:
        data = json.loads(text.strip())
        source = data["source"]
        if not isinstance(source, str) or not source.strip() or len(source) > max_chars:
            return None
        return TranslationTask(source.strip(), source_lang, target_lang)
    except (ValueError, TypeError, KeyError):
        return None


def learnability_rewards(successes: Tensor) -> Tensor:
    """AZR equation 4: zero if unsolved, otherwise one minus solver success rate.

    Rows correspond to proposed tasks and columns to sampled solver responses.
    Translation success is a thresholded, fixed feedback-model judgment.
    """
    if successes.ndim != 2 or successes.shape[1] < 2:
        raise ValueError("Learnability needs at least two solver samples per task")
    rate = successes.float().mean(-1)
    return torch.where(rate > 0, 1 - rate, torch.zeros_like(rate))


class TaskMemory:
    """A bounded, deduplicated buffer of validated tasks, serialized in checkpoints."""

    def __init__(self, capacity: int = 128) -> None:
        if capacity < 1:
            raise ValueError("Memory capacity must be positive")
        self.capacity = capacity
        self.tasks: list[TranslationTask] = []

    def add(self, task: TranslationTask) -> bool:
        """Add a new task, evicting the oldest when full."""
        key = (task.source.casefold(), task.source_lang, task.target_lang)
        if any(
            (t.source.casefold(), t.source_lang, t.target_lang) == key
            for t in self.tasks
        ):
            return False
        self.tasks.append(task)
        self.tasks = self.tasks[-self.capacity :]
        return True

    def state_dict(self) -> list[dict[str, str]]:
        """Return JSON-compatible task records."""
        return [
            {
                "source": t.source,
                "source_lang": t.source_lang,
                "target_lang": t.target_lang,
            }
            for t in self.tasks
        ]

    def load_state_dict(self, state: list[dict[str, str]]) -> None:
        """Validate and restore task records under the current capacity."""
        tasks: list[TranslationTask] = []
        for item in state:
            if any(
                not isinstance(item.get(key), str) or not item[key].strip()
                for key in ("source", "source_lang", "target_lang")
            ):
                raise ValueError("Invalid task memory checkpoint")
            tasks.append(TranslationTask(**item))
        self.tasks.clear()
        for task in tasks:
            self.add(task)
