"""Unsupported Claim Rate: fraction of a generated answer's claims that
failed grounding, for any reason.

An approximation, stated honestly: `GroundedResponse.generation_statistics`
exposes only the aggregate `claims_grounded`/`claims_total` counts, not a
breakdown by specific failure mode (unsupported vs. missing citation vs.
unresolved citation) -- exposing that breakdown would require adding new
fields to Module 10's own statistics model, which this module's zero-touch
principle rules out. This rate therefore measures "claims that failed
grounding for any reason," a coarser signal than a true per-status
unsupported-claim count (see the final report's known limitations).
"""

from backend.evaluation.exceptions import MetricComputationError


def unsupported_claim_rate(claims_grounded: int, claims_total: int) -> float:
    """Compute the unsupported claim rate for one generated answer.

    Args:
        claims_grounded: Number of claims that passed grounding validation.
        claims_total: Total claims extracted from the answer.

    Returns:
        `(claims_total - claims_grounded) / claims_total`, in `[0.0, 1.0]`.

    Raises:
        MetricComputationError: `claims_total` is not positive.
    """
    if claims_total <= 0:
        raise MetricComputationError(reason=f"claims_total must be positive, got {claims_total}")
    return (claims_total - claims_grounded) / claims_total
