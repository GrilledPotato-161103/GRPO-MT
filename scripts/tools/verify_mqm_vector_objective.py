"""Numerical reference for the proposed MQM vector objective, not a trainer.

Uses NumPy and finite differences; does not download a model or contact a judge.
Run from the project root with the existing system Python. The result verifies
the mathematics on synthetic data, not translation quality or token alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def vector_loss(
    logps: np.ndarray,
    masks: np.ndarray,
    weights: np.ndarray,
    *,
    beta: float = 1.0,
    preference_target: float = 0.9,
    anchor_weight: float = 0.1,
    nll_floor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return K losses and K x N x 2 x T log-probability coefficients.

    logps: N x 2 x T, with verified repair first, original second.
    masks: N x K x 2 x T, frozen, separately aligned on each full sequence.
    weights: N x K, frozen eligibility x empirical reliability x severity cost.
    A zero weight is abstention; an active contrast needs both nonempty masks.
    N includes abstained records: selection must not renormalize weak evidence.
    This is a local contrast surrogate, not a sequence-level DPO derivation.
    """
    logps = np.asarray(logps, dtype=float)
    masks = np.asarray(masks, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if logps.ndim != 3 or logps.shape[0] < 1 or logps.shape[1] != 2:
        raise ValueError("Need N x 2 x T log probabilities with N >= 1")
    n, _, t = logps.shape
    if masks.ndim != 4 or masks.shape[0] != n or masks.shape[2:] != (2, t):
        raise ValueError("Need aligned N x K x 2 x T masks")
    k = masks.shape[1]
    if k < 1 or weights.shape != (n, k):
        raise ValueError("Need N x K weights with K >= 1")
    if any(not np.isfinite(a).all() for a in (logps, masks, weights)):
        raise ValueError("All input arrays must be finite")
    if (logps > 0).any() or (weights < 0).any() or not np.isin(masks, [0, 1]).all():
        raise ValueError("Invalid log probabilities, nonnegative weights or binary masks")
    parameters = (beta, preference_target, anchor_weight, nll_floor)
    if not all(np.isfinite(x) for x in parameters):
        raise ValueError("Objective parameters must be finite")
    if beta <= 0 or not 0.5 < preference_target < 1:
        raise ValueError("Positive beta and a soft preference target in (.5, 1) required")
    if anchor_weight < 0 or nll_floor < 0:
        raise ValueError("Nonnegative anchor weight and NLL floor required")
    counts = masks.sum(-1)
    if ((counts == 0) & (weights[:, :, None] > 0)).any():
        raise ValueError("Active repair contrasts require both masks, including edit boundaries")
    normalized = masks / np.maximum(counts[..., None], 1)
    scores = (normalized * logps[:, None]).sum(-1)
    delta = scores[:, :, 0] - scores[:, :, 1]
    z = beta * delta
    # Stable BCE: q * softplus(-z) + (1-q) * softplus(z).
    bce = preference_target * np.logaddexp(0, -z) + (
        1 - preference_target
    ) * np.logaddexp(0, z)
    hinge = np.maximum(-scores[:, :, 0] - nll_floor, 0)
    terms = bce + 0.5 * anchor_weight * hinge**2
    losses = (weights * terms).sum(0) / n
    sigmoid = np.exp(-np.logaddexp(0, -z))
    contrast_coefficient = beta * (sigmoid - preference_target)
    coefficient = np.stack(
        [contrast_coefficient - anchor_weight * hinge, -contrast_coefficient],
        axis=2,
    )
    coefficient = weights[:, :, None, None] * coefficient[..., None] * normalized / n
    return losses, coefficient.transpose(1, 0, 2, 3)


def actor_logps(
    theta: np.ndarray, features: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Tiny shared softmax actor, with different full-prefix features per side."""
    logits = features @ theta
    log_normalizer = np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    all_logps = logits - log_normalizer
    selected = np.take_along_axis(all_logps, targets[..., None], axis=-1)[..., 0]
    return selected, np.exp(all_logps)


def actor_gradients(
    coefficients: np.ndarray,
    probabilities: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Chain rule: J(log p)^T c, preserving the criterion dimension."""
    one_hot = np.eye(probabilities.shape[-1])[targets]
    logits_grad = coefficients[..., None] * (one_hot - probabilities)[None]
    return np.einsum("nsth,knstv->khv", features, logits_grad)


def finite_difference(function, point: np.ndarray, step: float = 1e-6) -> np.ndarray:
    baseline = np.atleast_1d(function(point))
    gradient = np.empty((len(baseline), *point.shape))
    for index in np.ndindex(point.shape):
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        gradient[(slice(None), *index)] = (function(plus) - function(minus)) / (2 * step)
    return gradient


def project_displacement(
    proposed: np.ndarray, protected: np.ndarray, radius: float
) -> np.ndarray:
    """Small reference QP: min ||u-u0||^2 s.t. G u <= 0 and ||u|| <= radius.

    Enumerates active sets, suitable only for this tiny numerical demonstration.
    Inputs are actual parameter displacements, not raw gradients to feed Adam.
    The halfspaces form a cone, so radial clipping preserves feasibility and
    gives the Euclidean projection onto its intersection with a centered ball.
    """
    proposed, protected = np.asarray(proposed, float), np.asarray(protected, float)
    if proposed.ndim != 1 or protected.ndim != 2 or protected.shape[1] != proposed.size:
        raise ValueError("Need a displacement vector and aligned gradient rows")
    if not np.isfinite(proposed).all() or not np.isfinite(protected).all():
        raise ValueError("Displacement and gradients must be finite")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Need a positive finite radius")
    candidates = [np.zeros_like(proposed)]
    for size in range(protected.shape[0] + 1):
        for active in itertools.combinations(range(protected.shape[0]), size):
            if active:
                constraints = protected[list(active)]
                multiplier = np.linalg.lstsq(
                    constraints @ constraints.T, constraints @ proposed, rcond=None
                )[0]
                if (multiplier < -1e-10).any():
                    continue
                candidate = proposed - constraints.T @ multiplier
                if not np.allclose(constraints @ candidate, 0, atol=1e-9, rtol=0):
                    continue
            else:
                candidate = proposed.copy()
            if (protected @ candidate <= 1e-9).all():
                candidates.append(candidate)
    solution = min(candidates, key=lambda u: float(np.square(u - proposed).sum()))
    return solution * min(1.0, radius / max(float(np.linalg.norm(solution)), 1e-30))


def verify() -> dict:
    rng = np.random.default_rng(20261116)
    n, k, t, hidden, vocabulary = 3, 3, 5, 4, 7
    features = rng.normal(size=(n, 2, t, hidden))
    targets = rng.integers(vocabulary, size=(n, 2, t))
    theta = rng.normal(scale=0.2, size=(hidden, vocabulary))
    masks = np.zeros((n, k, 2, t))
    masks[0, 0, :, 0] = 1  # Orthography: initial token contrast.
    masks[0, 1, :, 2:4] = 1  # Wrong sense: local phrase contrast.
    masks[1, 2, 0, 1:4] = 1  # Omission: inserted tokens on repaired side.
    masks[1, 2, 1, 1] = 1  # First divergent continuation on original side.
    weights = np.array([[0.7, 0.6, 0.0], [0.0, 0.0, 0.8], [0.0, 0.0, 0.0]])
    logps, probabilities = actor_logps(theta, features, targets)
    losses, coefficients = vector_loss(logps, masks, weights)
    numeric_coefficients = finite_difference(lambda p: vector_loss(p, masks, weights)[0], logps)
    coefficient_error = float(np.abs(coefficients - numeric_coefficients).max())
    assert coefficient_error < 1e-8, coefficient_error
    gradients = actor_gradients(coefficients, probabilities, features, targets)
    numeric_actor = finite_difference(
        lambda w: vector_loss(actor_logps(w, features, targets)[0], masks, weights)[0], theta
    )
    actor_error = float(np.abs(gradients - numeric_actor).max())
    assert actor_error < 1e-8, actor_error

    # No direct coefficient on unaffected actions or abstained examples.
    support = masks.transpose(1, 0, 2, 3).astype(bool)
    assert np.count_nonzero(coefficients[~support]) == 0
    assert np.count_nonzero(coefficients[:, 2]) == 0
    empty_losses, empty_coefficient = vector_loss(logps, masks, np.zeros_like(weights))
    assert not empty_losses.any() and not empty_coefficient.any()
    assert not np.allclose(coefficients[0], coefficients[1])
    assert np.linalg.norm(gradients[2]) > 0  # Insertions supply a nonzero actor signal.

    # Dense versus source-microbatch chain rule; the denominator stays global N.
    microbatch_gradients = np.zeros_like(gradients)
    for source in range(n):
        _, local_coefficient = vector_loss(
            logps[source : source + 1], masks[source : source + 1], weights[source : source + 1]
        )
        microbatch_gradients += actor_gradients(
            local_coefficient / n,
            probabilities[source : source + 1],
            features[source : source + 1],
            targets[source : source + 1],
        )
    microbatch_error = float(np.abs(microbatch_gradients - gradients).max())
    assert microbatch_error < 1e-12

    # Identical masks reduce to a scalar coefficient. Different masks can give
    # opposite token coefficients even when their sum per completion is zero.
    criterion_advantages = np.array([[1.0, -1.0]])
    criterion_weights = np.array([1.0, 1.0])
    common_mask = np.array([[0.5, 0.5]])
    per_criterion = criterion_advantages[:, :, None] * common_mask[:, None, :]
    scalar = (criterion_advantages @ criterion_weights)[:, None] * common_mask
    assert np.allclose(np.einsum("k,nkt->nt", criterion_weights, per_criterion), scalar)
    localized = np.array([[[1.0, 0.0], [0.0, -1.0]]]).sum(1)
    assert localized.sum() == 0 and np.count_nonzero(localized) == 2

    # Unknown/misaligned targets cannot acquire an anchor by clipping empty masks.
    rejected_empty_active = False
    try:
        vector_loss(logps, np.zeros_like(masks), np.ones_like(weights))
    except ValueError:
        rejected_empty_active = True
    assert rejected_empty_active

    # With alpha=0 the soft preference has a finite stationary margin.
    optimum = np.log(0.9 / 0.1)
    stationary = np.exp(-np.logaddexp(0, -optimum)) - 0.9
    assert abs(stationary) < 1e-12

    # One pass of halfspace projections can re-violate a preceding constraint.
    protected = np.array([[1.0, 0.0], [-1.0, 1.0]])
    proposed = np.array([1.0, 1.0])
    sequential = proposed.copy()
    for g in protected:
        sequential -= max(float(g @ sequential), 0) * g / (g @ g)
    assert (protected @ sequential).max() > 0
    projected = project_displacement(proposed, protected, radius=1.0)
    assert (protected @ projected <= 1e-9).all()
    assert np.linalg.norm(projected) <= 1 + 1e-9
    assert np.allclose(projected, 0, atol=1e-9)
    nonzero_projection = project_displacement(
        np.array([1.0, 2.0]), np.array([[1.0, 0.0]]), radius=0.5
    )
    assert np.allclose(nonzero_projection, [0, 0.5])

    # A raw-gradient constraint does not survive arbitrary optimizer preconditioning.
    protected_gradient = np.array([1.0, 1.0])
    gradient_direction = np.array([2.0, -1.0])
    assert protected_gradient @ gradient_direction > 0
    preconditioned_update = -(np.diag([0.1, 10.0]) @ gradient_direction)
    assert protected_gradient @ preconditioned_update > 0

    # A zero first derivative does not constrain curvature or certify retention.
    zero_grad = np.array([[0.0, 0.0]])
    zero_protected = project_displacement(np.array([0.1, 0]), zero_grad, radius=1)
    assert np.square(zero_protected).sum() > 0  # L(theta)=||theta||^2 at theta=0.

    flat = gradients.reshape(k, -1)
    norms = np.linalg.norm(flat, axis=1)
    cosine = flat @ flat.T / np.maximum(norms[:, None] * norms[None], 1e-30)
    return {
        "schema_version": 1,
        "purpose": "Synthetic mathematics verification; not autograd, MT quality or integration validation",
        "seed": 20261116,
        "criteria": ["orthography", "mistranslation", "omission"],
        "checks": {
            "coefficient_finite_difference": True,
            "shared_actor_chain_rule_finite_difference": True,
            "criterion_specific_token_support": True,
            "abstention_is_zero_without_renormalization": True,
            "no_verified_contrasts_is_zero": True,
            "insertion_boundary_produces_signal": True,
            "dense_equals_source_microbatch": True,
            "fixed_scalarization_equivalence": True,
            "localized_coefficients_survive_scalar_cancellation": True,
            "empty_active_masks_rejected": True,
            "soft_preference_finite_stationary_margin": True,
            "single_pass_projection_counterexample": True,
            "joint_displacement_projection_feasible": True,
            "projection_with_nonzero_feasible_step": True,
            "optimizer_preconditioning_counterexample": True,
            "zero_gradient_is_not_retention_certificate": True,
        },
        "maximum_absolute_errors": {
            "logprob_coefficients": coefficient_error,
            "actor_parameter_gradients": actor_error,
            "dense_vs_microbatch": microbatch_error,
        },
        "synthetic_losses": losses.tolist(),
        "synthetic_gradient_cosines": cosine.tolist(),
        "projection_example": {
            "proposed_displacement": proposed.tolist(),
            "single_pass_displacement": sequential.tolist(),
            "joint_projected_displacement": projected.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("diagnostics/mqm_vector_20261116/verification.json"),
    )
    args = parser.parse_args()
    result = verify()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks": len(result["checks"]),
                      "maximum_absolute_errors": result["maximum_absolute_errors"]}, indent=2))


if __name__ == "__main__":
    main()
