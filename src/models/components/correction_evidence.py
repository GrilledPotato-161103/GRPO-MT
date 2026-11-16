"""Optional structured criterion evidence for noisy correction routing."""

import math
from typing import Any

from src.feedback.backends import APIChatBackend, TransformersChatBackend
from src.feedback.criteria import CriterionJudge
from src.feedback.judge import TranslationTask


class CorrectionEvidence:
    """Unknown is abstention, never a positive target or a negative label.

    `calibrated` is the caller's assertion of external validation. A judge's
    self-reported confidence is insufficient without that assertion.
    """

    required = frozenset(
        {
            "adequacy",
            "omission",
            "addition",
            "mistranslation",
            "negation",
            "numbers",
            "target_language",
            "fluency",
            "orthography",
        }
    )

    def __init__(
        self,
        enabled: bool = False,
        calibrated: bool = False,
        confidence_threshold: float = 0.9,
        judge: Any = None,
        backend: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        cache_only: bool = False,
    ):
        if not isinstance(enabled, bool) or not isinstance(calibrated, bool):
            raise TypeError("Evidence enabled/calibrated must be boolean")
        if (
            isinstance(confidence_threshold, bool)
            or not math.isfinite(confidence_threshold)
            or not 0 <= confidence_threshold <= 1
        ):
            raise ValueError("Invalid evidence confidence threshold")
        self.enabled, self.calibrated = enabled, calibrated
        self.confidence_threshold = confidence_threshold
        self.judge = judge
        if enabled and judge is None:
            options = dict(backend or {})
            kind = options.pop("kind", "api")
            if kind == "api":
                options.setdefault("max_tokens", 2048)
                evaluator = APIChatBackend(**options)
            elif kind == "self_hosted":
                evaluator = TransformersChatBackend(**options)
            else:
                raise ValueError("Unknown correction evidence backend")
            self.judge = CriterionJudge(evaluator, cache_dir, cache_only)

    def assess(
        self,
        sources: list[str],
        candidates: list[str],
        source_langs: list[str],
        target_langs: list[str],
    ) -> list[str]:
        if any(
            len(v) != len(sources) for v in (candidates, source_langs, target_langs)
        ):
            raise ValueError("Evidence inputs must be aligned")
        if not self.enabled:
            return ["pass"] * len(sources)
        tasks = [
            TranslationTask(s, a, b)
            for s, a, b in zip(sources, source_langs, target_langs, strict=True)
        ]
        judgments = self.judge.evaluate(tasks, candidates)
        if len(judgments) != len(sources):
            raise ValueError("Criterion feedback lost candidate alignment")
        result = []
        for criteria in judgments:
            known = {
                c.name: c
                for c in criteria
                if c.name in self.required
                and (
                    (
                        c.name == "numbers"
                        and not c.applicable
                        and c.value is None
                        and c.severity == "none"
                    )
                    or (
                        c.applicable
                        and c.value is not None
                        and c.confidence is not None
                        and c.confidence >= self.confidence_threshold
                    )
                )
            }
            if not self.calibrated:
                result.append("unknown")
            elif any(c.severity != "none" and c.evidence for c in known.values()):
                result.append("reject")
            elif self.required.issubset(known) and all(
                c.severity == "none" and (not c.applicable or c.value >= 0.8)
                for c in known.values()
            ):
                result.append("pass")
            else:
                result.append("unknown")
        return result
