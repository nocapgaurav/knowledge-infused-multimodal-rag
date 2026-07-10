"""Unit tests for Precision@K."""

import pytest

from backend.evaluation.exceptions import MetricComputationError
from backend.evaluation.metrics.precision import precision_at_k


def test_precision_at_k_counts_hits_in_top_k() -> None:
    retrieved = ["x", "a", "b"]
    relevant = {"a", "b"}

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_precision_at_k_ignores_items_beyond_cutoff() -> None:
    retrieved = ["a", "x", "x", "b"]
    relevant = {"a", "b"}

    assert precision_at_k(retrieved, relevant, 1) == pytest.approx(1.0)


def test_precision_at_k_penalizes_short_result_sets() -> None:
    retrieved = ["a"]
    relevant = {"a", "b"}

    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(1 / 5)


def test_precision_at_k_is_zero_when_nothing_relevant_retrieved() -> None:
    retrieved = ["x", "y", "z"]
    relevant = {"a", "b"}

    assert precision_at_k(retrieved, relevant, 3) == 0.0


def test_precision_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(MetricComputationError):
        precision_at_k(["a"], {"a"}, 0)
