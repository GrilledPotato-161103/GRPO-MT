"""Unknown, contradictory and surface-quality feedback in imperfect environments."""

from dataclasses import replace

from src.feedback.criteria import PROMPT_VERSION, Criterion
from src.models.components.correction_evidence import CorrectionEvidence


class Judge:
    def __init__(self, criteria):
        self.criteria = criteria

    def evaluate(self, tasks, candidates):
        return [self.criteria] * len(tasks)


def assess(criteria, calibrated=True):
    evidence = CorrectionEvidence(
        enabled=True, calibrated=calibrated, judge=Judge(criteria)
    )
    return evidence.assess(["source"], ["target"], ["English"], ["Vietnamese"])[0]


def complete():
    return tuple(
        Criterion(n, True, 0.95, 0.95, "none", ()) for n in CorrectionEvidence.required
    )


def test_full_sac_cannot_override_independent_fluency_or_capitalization_error():
    assert PROMPT_VERSION == "mt-criteria-v2"
    for name in ("fluency", "orthography"):
        criteria = tuple(
            replace(c, value=0.5, severity="minor", evidence=("observed error",))
            if c.name == name
            else c
            for c in complete()
        )
        assert assess(criteria) == "reject"


def test_missing_unknown_and_uncalibrated_evidence_never_certifies_an_anchor():
    assert assess(complete()) == "pass"
    assert assess(complete()[:-1]) == "unknown"
    assert assess(complete(), calibrated=False) == "unknown"
    assert (
        assess(
            tuple(
                replace(c, confidence=None) if c.name == "fluency" else c
                for c in complete()
            )
        )
        == "unknown"
    )
    assert (
        assess(
            tuple(
                replace(c, value=0.2) if c.name == "fluency" else c for c in complete()
            )
        )
        == "unknown"
    )


def test_numbers_may_be_explicitly_inapplicable_without_fabricating_a_score():
    criteria = tuple(
        replace(c, applicable=False, value=None, confidence=None)
        if c.name == "numbers"
        else c
        for c in complete()
    )
    assert assess(criteria) == "pass"
