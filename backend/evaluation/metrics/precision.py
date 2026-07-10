"""Precision@K: of the top K retrieved items, what fraction are relevant.

The conventional IR definition divides by `k` itself, not by however many
items were actually retrieved -- a result set shorter than `k` is
penalized for the missing slots rather than scored as if `k` had been
smaller.
"""

from collections.abc import Sequence

from backend.evaluation.exceptions import MetricComputationError


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Compute Precision@K.

    Args:
        retrieved_ids: Retrieved item ids, best match first.
        relevant_ids: The ground-truth set of relevant item ids.
        k: Cutoff rank.

    Returns:
        Precision@K in `[0.0, 1.0]`.

    Raises:
        MetricComputationError: `k` is not positive.
    """
    if k <= 0:
        raise MetricComputationError(reason=f"k must be positive, got {k}")
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return hits / k
