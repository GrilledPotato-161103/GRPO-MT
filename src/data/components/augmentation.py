"""Online, aligned translation augmentation without held-out or cross-stage data."""

import math
import random
import re
import time
import unicodedata
from typing import Any


class TranslationAugmentation:
    """Perturb source formatting and optionally concatenate two aligned pairs.

    Word deletion, accent removal, number changes, and unverified paraphrases
    are deliberately excluded. Concatenation uses the caller's current split.
    """

    def __init__(
        self,
        whitespace_probability: float = 0.2,
        unicode_probability: float = 0.2,
        concat_probability: float = 0.1,
        max_concat_chars: int = 768,
    ) -> None:
        for value in (whitespace_probability, unicode_probability, concat_probability):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    "Augmentation probabilities must be finite and in [0,1]"
                )
        if max_concat_chars < 1:
            raise ValueError("max_concat_chars must be positive")
        self.whitespace_probability = whitespace_probability
        self.unicode_probability = unicode_probability
        self.concat_probability = concat_probability
        self.max_concat_chars = max_concat_chars

    def apply(
        self,
        source: str,
        reference: str,
        partner: tuple[str, str] | None,
        rng: random.Random,
    ) -> dict[str, str]:
        """Return aligned text with a clean source retained for semantic rewards."""
        started = time.perf_counter()
        operations = []
        if partner is not None and rng.random() < self.concat_probability:
            joined_source, joined_reference = (
                source + " " + partner[0],
                reference + " " + partner[1],
            )
            if max(len(joined_source), len(joined_reference)) <= self.max_concat_chars:
                source, reference = joined_source, joined_reference
                operations.append("concat")
        clean_source = source
        if rng.random() < self.unicode_probability:
            source = unicodedata.normalize(rng.choice(("NFC", "NFD")), source)
            operations.append("unicode")
        if rng.random() < self.whitespace_probability:
            source = re.sub(
                r"\s+", lambda _: rng.choice((" ", "  ", "\t", "\n")), source
            )
            source = rng.choice(("", " ", "\t")) + source + rng.choice(("", " ", "\n"))
            operations.append("whitespace")
        return {
            "source": source,
            "reference": reference,
            "clean_source": clean_source,
            "augmentation": "+".join(operations) or "none",
            "augmentation_seconds": str(time.perf_counter() - started),
        }


def make_augmentation(options: dict[str, Any] | None) -> TranslationAugmentation | None:
    """Construct augmentation only when explicitly enabled for training."""
    if not options or not options.get("enabled", False):
        return None
    return TranslationAugmentation(
        **{key: value for key, value in options.items() if key != "enabled"}
    )
