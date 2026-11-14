"""COMET/SAC curriculum fusion and compatible legacy translation rewards."""

import math
import time
from typing import Any

from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF

from src.models.components.comet_reward import CometScorer
from src.models.components.reward_fusion import DynamicRewardFusion
from src.models.components.signed_sac import SignedSAC

SAC_METRICS = (
    "sac",
    "round_trip_sac",
    "round_trip_penalty",
    "sac_gated_comet",
    "endpoint_sac",
    "symmetric_sac",
    "symmetric_penalty",
    "comet_reward",
    "sac_reward",
    "fused_comet_score",
    "fused_sac_score",
    "fusion_alpha",
    "fusion_normalized",
    "sac_positive_reward",
    "sac_penalty_severity",
    "sac_match_mask",
    "sac_penalty_gate",
    "sac_positive_fused_reward",
    "sac_negative_reward",
    "sac_signed_reward",
)


class TranslationReward:
    """Combine translation quality and faithfulness as higher-is-better rewards."""

    def __init__(
        self,
        chrf_weight: float = 1.0,
        bleu_weight: float = 0.0,
        comet_weight: float = 0.0,
        comet_model_name: str = "Unbabel/wmt22-cometkiwi-da",
        comet_device: str = "cpu",
        comet_batch_size: int = 8,
        comet_cache_dir: str | None = None,
        comet_scorer: Any = None,
        comet_api_key_env: str = "HUGGING_FACE_API_KEY",
        harness: dict[str, Any] | None = None,
        sac: dict[str, Any] | None = None,
        sac_scorer: Any = None,
    ) -> None:
        weights = (chrf_weight, bleu_weight, comet_weight)
        if (
            any(not math.isfinite(weight) or weight < 0 for weight in weights)
            or sum(weights) <= 0
        ):
            raise ValueError("Reward weights must be nonnegative with a positive sum")
        self.chrf_weight = chrf_weight
        self.bleu_weight = bleu_weight
        self.comet_weight = comet_weight
        options = dict(sac or {})
        self.sac_enabled = options.pop("enabled", False)
        self.fusion = DynamicRewardFusion(**options.pop("fusion", {}))
        self.polarity = SignedSAC(**options.pop("polarity", {}))
        if self.polarity.enabled and (
            not self.fusion.enabled
            or not self.sac_enabled
            or self.fusion.sac_direct_weight != 1
        ):
            raise ValueError(
                "Signed SAC requires direct-only categorical SAC and enabled COMET/SAC fusion"
            )
        if self.fusion.enabled and not self.sac_enabled:
            raise ValueError("Dynamic COMET/SAC fusion requires SAC enabled")
        self.consistency_mode = options.pop("consistency_mode", "round_trip")
        self.round_trip_weight = options.pop("round_trip_weight", 0.2)
        self.consistency_weight = options.pop(
            "consistency_weight", self.round_trip_weight
        )
        if self.consistency_mode not in {"round_trip", "symmetric_endpoints"}:
            raise ValueError(
                "SAC consistency_mode must be round_trip or symmetric_endpoints"
            )
        if not math.isfinite(self.consistency_weight) or self.consistency_weight < 0:
            raise ValueError("SAC consistency_weight must be finite and nonnegative")
        self.round_trip_batch_size = options.pop("round_trip_batch_size", 8)
        self.round_trip_max_new_tokens = options.pop("round_trip_max_new_tokens", 256)
        if not math.isfinite(self.round_trip_weight) or self.round_trip_weight < 0:
            raise ValueError("SAC round_trip_weight must be finite and nonnegative")
        if (
            isinstance(self.round_trip_batch_size, bool)
            or not isinstance(self.round_trip_batch_size, int)
            or self.round_trip_batch_size < 1
        ):
            raise ValueError("SAC round_trip_batch_size must be a positive integer")
        if (
            isinstance(self.round_trip_max_new_tokens, bool)
            or not isinstance(self.round_trip_max_new_tokens, int)
            or self.round_trip_max_new_tokens < 1
        ):
            raise ValueError("SAC round_trip_max_new_tokens must be a positive integer")
        if self.sac_enabled and (comet_weight <= 0 or bleu_weight or chrf_weight):
            raise ValueError(
                "SAC reward requires COMET and zero BLEU/chrF reward weights"
            )
        if self.sac_enabled and (harness or {}).get("enabled"):
            raise ValueError(
                "SAC reward and the conditional reward harness are separate experiments"
            )
        from src.feedback.sac import SACScorer

        self.sac = (sac_scorer or SACScorer(**options)) if self.sac_enabled else None
        self.requires_sources = (
            comet_weight > 0 or bool((harness or {}).get("enabled")) or self.sac_enabled
        )
        from src.models.components.reward_harness import ConditionalRewardHarness

        self.harness = ConditionalRewardHarness(**(harness or {}))
        self.comet = comet_scorer or (
            CometScorer(
                comet_model_name,
                comet_device,
                comet_batch_size,
                comet_cache_dir,
                api_key_env=comet_api_key_env,
            )
            if comet_weight > 0
            else None
        )
        self.chrf = CHRF(word_order=2)
        self.bleu = BLEU(effective_order=True)
        self.last_timings: dict[str, float] = {}

    @property
    def requires_consistency(self) -> bool:
        """Only compute the second SAC comparison if its active objective uses it."""
        return (
            self.fusion.sac_direct_weight < 1
            if self.fusion.enabled
            else bool(self.consistency_weight)
        )

    def components(
        self,
        predictions: list[str],
        references: list[str],
        *,
        sources: list[str] | None = None,
        source_langs: list[str] | None = None,
        target_langs: list[str] | None = None,
        apply_harness: bool = True,
        reconstructions: list[str] | None = None,
        partner_indices: list[int] | None = None,
        optimizer_step: int = 0,
        candidate_group_size: int = 1,
    ) -> dict[str, list[float]]:
        """Score each candidate against its matching reference translation."""
        if len(predictions) != len(references):
            raise ValueError("Each prediction needs exactly one reference")
        started = time.perf_counter()
        chrf = [
            self.chrf.sentence_score(p, [r]).score / 100 if p.strip() else 0.0
            for p, r in zip(predictions, references, strict=True)
        ]
        chrf_seconds = time.perf_counter() - started
        started = time.perf_counter()
        bleu = [
            self.bleu.sentence_score(p, [r]).score / 100 if p.strip() else 0.0
            for p, r in zip(predictions, references, strict=True)
        ]
        components = {"chrf": chrf, "bleu": bleu}
        self.last_timings = {
            "chrf": chrf_seconds,
            "bleu": time.perf_counter() - started,
        }
        if self.comet_weight > 0:
            if sources is None or len(sources) != len(predictions):
                raise ValueError("COMETKiwi reward requires aligned source texts")
            assert self.comet is not None
            started = time.perf_counter()
            raw = list(self.comet(sources, predictions, references))
            self.last_timings["cometkiwi"] = time.perf_counter() - started
            for phase, seconds in getattr(self.comet, "last_timings", {}).items():
                self.last_timings[f"cometkiwi_{phase}"] = seconds
            if len(raw) != len(predictions) or any(
                not math.isfinite(value) for value in raw
            ):
                raise ValueError("COMET must return one finite score per candidate")
            components["comet"] = raw
        total = self.chrf_weight + self.bleu_weight + self.comet_weight
        components["reward"] = [
            (
                self.chrf_weight * components["chrf"][i]
                + self.bleu_weight * components["bleu"][i]
                + self.comet_weight
                * min(
                    1.0, max(0.0, components.get("comet", [0.0] * len(predictions))[i])
                )
            )
            / total
            if prediction.strip()
            else 0.0
            for i, prediction in enumerate(predictions)
        ]
        if self.sac_enabled:
            count = len(predictions)
            if (
                sources is None
                or source_langs is None
                or target_langs is None
                or any(
                    len(values) != count
                    for values in (sources, source_langs, target_langs)
                )
            ):
                raise ValueError(
                    "SAC reward requires aligned sources and both language labels"
                )
            need_consistency = self.requires_consistency
            round_trip = self.consistency_mode == "round_trip" and need_consistency
            endpoints = (
                self.consistency_mode == "symmetric_endpoints" and need_consistency
            )
            if round_trip and (
                reconstructions is None or len(reconstructions) != count
            ):
                raise ValueError(
                    "Round-trip SAC requires one reconstruction per candidate"
                )
            if endpoints:
                if (
                    partner_indices is None
                    or len(partner_indices) != count
                    or any(
                        isinstance(other, bool)
                        or not isinstance(other, int)
                        or not 0 <= other < count
                        or other == i
                        for i, other in enumerate(partner_indices)
                    )
                ):
                    raise ValueError(
                        "Symmetric SAC requires one distinct partner per candidate"
                    )
                if any(
                    partner_indices[other] != i
                    or source_langs[i] != target_langs[other]
                    or target_langs[i] != source_langs[other]
                    or source_langs[i] == target_langs[i]
                    or sources[i] != references[other]
                    or references[i] != sources[other]
                    for i, other in enumerate(partner_indices)
                ):
                    raise ValueError(
                        "Symmetric SAC partner texts and languages must be reversed pairs"
                    )
            assert self.sac is not None
            started = time.perf_counter()
            extra = round_trip or endpoints
            scores = self.sac.evaluate(
                sources + (references if endpoints else sources if round_trip else []),
                predictions
                + (
                    predictions
                    if endpoints
                    else (reconstructions or [])
                    if round_trip
                    else []
                ),
                source_langs
                + (target_langs if endpoints else source_langs if round_trip else []),
                target_langs
                + (target_langs if endpoints else source_langs if round_trip else []),
            )
            if len(scores) != count * (2 if extra else 1) or any(
                not math.isfinite(value) or value not in {0.1, 0.3, 0.7, 1.0}
                for value in scores
            ):
                raise ValueError("SAC must return one valid tier score per comparison")
            self.last_timings["sac"] = time.perf_counter() - started
            components["sac"] = scores[:count]
            if round_trip:
                components["round_trip_sac"] = scores[count:]
                if not self.fusion.enabled:
                    components["round_trip_penalty"] = [
                        self.consistency_weight * (1 - score)
                        for score in scores[count:]
                    ]
            if endpoints:
                assert partner_indices is not None
                components["endpoint_sac"] = scores[count:]
                components["symmetric_sac"] = [
                    (scores[count + i] + scores[count + other]) / 2
                    for i, other in enumerate(partner_indices)
                ]
                if not self.fusion.enabled:
                    components["symmetric_penalty"] = [
                        self.consistency_weight * (1 - score)
                        for score in components["symmetric_sac"]
                    ]
            components["sac_gated_comet"] = [
                quality * score
                for quality, score in zip(
                    components["reward"], scores[:count], strict=True
                )
            ]
            components["reward"] = [
                quality - penalty
                for quality, penalty in zip(
                    components["sac_gated_comet"],
                    components.get(
                        "symmetric_penalty",
                        components.get("round_trip_penalty", [0.0] * count),
                    ),
                    strict=True,
                )
            ]
            if self.fusion.enabled:
                consistency = components.get(
                    "symmetric_sac", components.get("round_trip_sac", components["sac"])
                )
                direct_weight = self.fusion.sac_direct_weight
                alignment = [
                    direct_weight * direct + (1 - direct_weight) * shared
                    if prediction.strip()
                    else 0.0
                    for prediction, direct, shared in zip(
                        predictions, components["sac"], consistency, strict=True
                    )
                ]
                quality = [
                    min(1.0, max(0.0, value)) if prediction.strip() else 0.0
                    for prediction, value in zip(
                        predictions, components["comet"], strict=True
                    )
                ]
                if self.polarity.enabled:
                    alignment, _, _ = self.polarity.separate(components["sac"])
                components.update(
                    self.fusion.apply(
                        quality, alignment, optimizer_step, candidate_group_size
                    )
                )
                if self.polarity.enabled:
                    components.update(
                        self.polarity.apply(components["sac"], components["reward"])
                    )
        if self.harness.enabled and apply_harness:
            if sources is None or source_langs is None or target_langs is None:
                raise ValueError(
                    "Conditional reward requires source text and both language labels"
                )
            started = time.perf_counter()
            components = self.harness.apply(
                components, predictions, sources, source_langs, target_langs
            )
            self.last_timings["structured_feedback"] = time.perf_counter() - started
        return components

    def __call__(
        self,
        predictions: list[str],
        references: list[str],
        *,
        sources: list[str] | None = None,
        source_langs: list[str] | None = None,
        target_langs: list[str] | None = None,
        reconstructions: list[str] | None = None,
        partner_indices: list[int] | None = None,
        optimizer_step: int = 0,
        candidate_group_size: int = 1,
    ) -> list[float]:
        """Return fused rewards or the configured legacy reward objective."""
        return self.components(
            predictions,
            references,
            sources=sources,
            source_langs=source_langs,
            target_langs=target_langs,
            reconstructions=reconstructions,
            partner_indices=partner_indices,
            optimizer_step=optimizer_step,
            candidate_group_size=candidate_group_size,
        )["reward"]
