"""NDCG@K: Normalized Discounted Cumulative Gain, with binary relevance.

Ground truth here is a set of relevant ids, not graded relevance scores,
so gain is binary (1 if relevant, 0 otherwise) -- this is the standard
"binary NDCG" formulation, and reduces to comparing the actual ranking's
discounted gain against the best-possible ranking's.
"""

import math
from collections.abc import Sequence

from backend.evaluation.exceptions import MetricComputationError


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Compute NDCG@K with binary relevance.

    Args:
        retrieved_ids: Retrieved item ids, best match first.
        relevant_ids: The ground-truth set of relevant item ids.
        k: Cutoff rank.

    Returns:
        NDCG@K in `[0.0, 1.0]`. `0.0` if no relevant items exist at all
        within the ideal top-`k` (i.e. `relevant_ids` is empty).

    Raises:
        MetricComputationError: `k` is not positive.
    """
    if k <= 0:
        raise MetricComputationError(reason=f"k must be positive, got {k}")

    top_k = retrieved_ids[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(top_k, start=1)
        if item in relevant_ids
    )

    ideal_hit_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hit_count + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
