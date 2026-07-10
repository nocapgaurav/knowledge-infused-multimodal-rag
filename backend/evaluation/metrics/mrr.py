"""Reciprocal Rank: the basis of Mean Reciprocal Rank.

Named `mrr.py` to match the requested package structure, but this module
computes one case's *reciprocal rank* -- the "mean" in MRR is the average
of this value across every case in a benchmark run, computed once during
aggregation (`EvaluationSummary.hybrid_retrieval_aggregate["mrr"]`), not
inside this function.
"""

from collections.abc import Sequence


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Compute one case's reciprocal rank.

    Args:
        retrieved_ids: Retrieved item ids, best match first.
        relevant_ids: The ground-truth set of relevant item ids.

    Returns:
        `1 / rank` of the first relevant item found, or `0.0` if none of
        the retrieved items are relevant.
    """
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0
