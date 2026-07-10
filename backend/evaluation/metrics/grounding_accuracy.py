"""Grounding Accuracy: fraction of a generated answer's claims that passed
Module 10's own Grounding Validation.

Derived entirely from `GroundedResponse.generation_statistics` -- Module
10 has already computed this deterministically (no LLM self-judgment);
this module reads that result rather than recomputing it, since
recomputing would mean duplicating grounding logic that could drift from
what production actually did.
"""

from backend.evaluation.exceptions import MetricComputationError


def grounding_accuracy(claims_grounded: int, claims_total: int) -> float:
    """Compute grounding accuracy for one generated answer.

    Args:
        claims_grounded: Number of claims that passed grounding validation
            (`GenerationStatistics.claims_grounded`).
        claims_total: Total claims extracted from the answer
            (`GenerationStatistics.claims_total`).

    Returns:
        Grounding accuracy in `[0.0, 1.0]`.

    Raises:
        MetricComputationError: `claims_total` is not positive.
    """
    if claims_total <= 0:
        raise MetricComputationError(reason=f"claims_total must be positive, got {claims_total}")
    return claims_grounded / claims_total
