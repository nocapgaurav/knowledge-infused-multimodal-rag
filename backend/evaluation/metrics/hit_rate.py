"""Hit Rate@K: whether at least one relevant item appears in the top K retrieved."""

from collections.abc import Sequence

from backend.evaluation.exceptions import MetricComputationError


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Compute Hit Rate@K.

    Args:
        retrieved_ids: Retrieved item ids, best match first.
        relevant_ids: The ground-truth set of relevant item ids.
        k: Cutoff rank.

    Returns:
        `1.0` if any of the top `k` retrieved items is relevant, else `0.0`.

    Raises:
        MetricComputationError: `k` is not positive.
    """
    if k <= 0:
        raise MetricComputationError(reason=f"k must be positive, got {k}")
    top_k = retrieved_ids[:k]
    return 1.0 if any(item in relevant_ids for item in top_k) else 0.0
