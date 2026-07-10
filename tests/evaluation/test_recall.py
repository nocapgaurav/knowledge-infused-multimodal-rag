"""Unit tests for Recall@K."""

import pytest

from backend.evaluation.exceptions import MetricComputationError
from backend.evaluation.metrics.recall import recall_at_k


def test_recall_at_k_counts_fraction_of_relevant_found() -> None:
    retrieved = ["x", "a", "b"]
    relevant = {"a", "b"}

    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1.0)


def test_recall_at_k_is_partial_when_cutoff_misses_some_relevant_items() -> None:
    retrieved = ["a", "x", "b"]
    relevant = {"a", "b"}

    assert recall_at_k(retrieved, relevant, 1) == pytest.approx(0.5)


def test_recall_at_k_is_zero_when_nothing_relevant_retrieved() -> None:
    retrieved = ["x", "y"]
    relevant = {"a", "b"}

    assert recall_at_k(retrieved, relevant, 2) == 0.0


def test_recall_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(MetricComputationError):
        recall_at_k(["a"], {"a"}, 0)


def test_recall_at_k_rejects_empty_relevant_set() -> None:
    with pytest.raises(MetricComputationError):
        recall_at_k(["a"], set(), 3)
