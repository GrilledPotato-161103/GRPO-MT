"""Conditional faithfulness constraints and post-normalization group confidence."""

import math
from dataclasses import asdict
from typing import Any

from src.feedback.backends import APIChatBackend, ChatBackend, TransformersChatBackend
from src.feedback.criteria import CriterionJudge
from src.feedback.judge import TranslationTask


class ConditionalRewardHarness:
    """Rank evidenced semantic failures below non-failures; preserve a fixed audit score.

    Judge confidence is an uncalibrated feature until calibrated externally.
    Requiring calibration evidence defaults to disabling confidence-based vetoes.
    Unknown criteria never become a fabricated passing judgment. All-fail groups
    receive zero policy-advantage weight and are recorded for verified repair.
    """

    def __init__(
        self,
        enabled: bool = False,
        judge: Any = None,
        backend: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        cache_only: bool = False,
        confidence_threshold: float = 0.9,
        calibrated: bool = False,
        unknown_group_weight: float = 0.25,
        semantic_weight: float = 0.5,
        lexical_relaxation: float = 0.5,
    ):
        for name, value in (
            ("confidence_threshold", confidence_threshold),
            ("unknown_group_weight", unknown_group_weight),
            ("semantic_weight", semantic_weight),
            ("lexical_relaxation", lexical_relaxation),
        ):
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be finite in [0,1]")
        self.enabled, self.calibrated = enabled, calibrated
        self.confidence_threshold, self.unknown_group_weight = (
            confidence_threshold,
            unknown_group_weight,
        )
        self.semantic_weight, self.lexical_relaxation = (
            semantic_weight,
            lexical_relaxation,
        )
        self.judge = judge
        if enabled and judge is None:
            evaluator: ChatBackend
            options = dict(backend or {})
            kind = options.pop("kind", "api")
            if kind == "api":
                options.setdefault("max_tokens", 2048)
                evaluator = APIChatBackend(**options)
            elif kind == "self_hosted":
                evaluator = TransformersChatBackend(**options)
            else:
                raise ValueError("Feedback kind must be api or self_hosted")
            self.judge = CriterionJudge(evaluator, cache_dir, cache_only)
        self.last_evidence: list[dict[str, Any]] = []

    def check_ready(self) -> None:
        if self.enabled and hasattr(self.judge, "check_ready"):
            self.judge.check_ready()

    def apply(
        self,
        components: dict[str, list[float]],
        predictions: list[str],
        sources: list[str],
        source_langs: list[str],
        target_langs: list[str],
    ) -> dict[str, list[float]]:
        result = dict(components)
        if not self.enabled:
            self.last_evidence = []
            return result
        count = len(predictions)
        if any(
            len(values) != count for values in (sources, source_langs, target_langs)
        ):
            raise ValueError(
                "Conditional feedback requires aligned texts and directions"
            )
        tasks = [
            TranslationTask(source, source_lang, target_lang)
            for source, source_lang, target_lang in zip(
                sources, source_langs, target_langs, strict=True
            )
        ]
        judgments = self.judge.evaluate(tasks, predictions)
        if len(judgments) != count:
            raise ValueError("Criterion feedback lost candidate alignment")
        result["audit_reward"] = list(components["reward"])
        for name in (
            "constraint_failure",
            "semantic_known",
            "group_confidence",
            "lexical_relaxed",
        ):
            result[name] = []
        result["reward"] = []
        self.last_evidence = []
        semantic_names = {
            "adequacy",
            "omission",
            "addition",
            "mistranslation",
            "negation",
        }
        for i, criteria in enumerate(judgments):
            semantic = [
                item
                for item in criteria
                if item.name in semantic_names
                and item.applicable
                and item.value is not None
            ]
            # A partial response cannot certify absence of a missing error category.
            complete = semantic_names.issubset(item.name for item in semantic)
            reliable = (
                self.calibrated
                and complete
                and all(
                    item.confidence is not None
                    and item.confidence >= self.confidence_threshold
                    for item in semantic
                )
            )
            protected = semantic_names | {"numbers", "target_language"}
            error = self.calibrated and (
                not predictions[i].strip()
                or any(
                    item.name in protected
                    and item.applicable
                    and item.value is not None
                    and item.confidence is not None
                    and item.confidence >= self.confidence_threshold
                    and item.severity in {"major", "critical"}
                    for item in criteria
                )
            )
            quality = components["reward"][i]
            relaxed = (
                reliable
                and not error
                and all(item.severity == "none" for item in semantic)
                and all(
                    item.value is not None and item.value >= self.confidence_threshold
                    for item in semantic
                )
            )
            if reliable:
                fidelity = min(
                    float(item.value) for item in semantic if item.value is not None
                )
                quality = (
                    1 - self.semantic_weight
                ) * quality + self.semantic_weight * fidelity
                if relaxed and "comet" in components:
                    quality = (
                        1 - self.lexical_relaxation
                    ) * quality + self.lexical_relaxation * min(
                        1.0, max(0.0, components["comet"][i])
                    )
            # Bounded tiers are non-compensatory for evidenced major/critical errors.
            # Even a high fluent score in the failure tier stays below the valid tier.
            result["reward"].append(-1.0 + 0.1 * quality if error else quality)
            result["constraint_failure"].append(float(error))
            result["semantic_known"].append(float(reliable))
            result["group_confidence"].append(
                1.0 if reliable or not self.calibrated else self.unknown_group_weight
            )
            result["lexical_relaxed"].append(float(relaxed))
            self.last_evidence.append(
                {
                    "candidate": i,
                    "reliable": reliable,
                    "failure": error,
                    "scorer": getattr(
                        self.judge, "scorer", {"class": type(self.judge).__qualname__}
                    ),
                    "criteria": [asdict(item) for item in criteria],
                }
            )
        return result


def group_update_weights(
    components: dict[str, list[float]], group_size: int, candidates: int
) -> list[float]:
    """Apply group confidence after GRPO normalization; never invent repair labels."""
    confidence = components.get("group_confidence", [1.0] * candidates)
    failures = components.get("constraint_failure", [0.0] * candidates)
    if (
        candidates % group_size
        or len(confidence) != candidates
        or len(failures) != candidates
    ):
        raise ValueError("Update weights require aligned complete candidate groups")
    result = []
    for offset in range(0, candidates, group_size):
        all_failed = all(v == 1 for v in failures[offset : offset + group_size])
        weight = 0.0 if all_failed else min(confidence[offset : offset + group_size])
        result.extend([weight] * group_size)
    return result
