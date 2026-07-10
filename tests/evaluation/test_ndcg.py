"""Unit tests for NDCG@K with binary relevance."""

import math

import pytest

from backend.evaluation.exceptions import MetricComputationError
from backend.evaluation.metrics.ndcg import ndcg_at_k


def test_ndcg_at_k_of_ideal_ranking_is_one() -> None:
    retrieved = ["a", "b"]
    relevant = {"a", "b"}

    assert ndcg_at_k(retrieved, relevant, 2) == pytest.approx(1.0)


def test_ndcg_at_k_penalizes_a_relevant_item_ranked_lower() -> None:
    retrieved = ["x", "a", "b"]
    relevant = {"a", "b"}

    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = dcg / idcg

    result = ndcg_at_k(retrieved, relevant, 3)

    assert result == pytest.approx(expected)
    assert result == pytest.approx(0.6934, abs=1e-3)


def test_ndcg_at_k_is_zero_when_relevant_set_is_empty() -> None:
    assert ndcg_at_k(["a", "b"], set(), 2) == 0.0


def test_ndcg_at_k_is_zero_when_nothing_relevant_retrieved() -> None:
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_ndcg_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(MetricComputationError):
        ndcg_at_k(["a"], {"a"}, 0)
