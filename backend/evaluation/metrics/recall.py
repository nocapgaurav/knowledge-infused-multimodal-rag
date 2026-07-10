"""Recall@K: of all relevant items, what fraction appear in the top K retrieved."""

from collections.abc import Sequence

from backend.evaluation.exceptions import MetricComputationError


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@K.

    Args:
        retrieved_ids: Retrieved item ids, best match first.
        relevant_ids: The ground-truth set of relevant item ids. Never
            empty in practice -- `EvaluationCase.expected_knowledge_units`
            is validated as non-empty at dataset load time.
        k: Cutoff rank.

    Returns:
        Recall@K in `[0.0, 1.0]`.

    Raises:
        MetricComputationError: `k` is not positive, or `relevant_ids` is empty.
    """
    if k <= 0:
        raise MetricComputationError(reason=f"k must be positive, got {k}")
    if not relevant_ids:
        raise MetricComputationError(reason="relevant_ids must not be empty")
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return hits / len(relevant_ids)
