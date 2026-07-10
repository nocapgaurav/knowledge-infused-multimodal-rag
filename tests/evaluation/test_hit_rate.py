"""Unit tests for Hit Rate@K."""

import pytest

from backend.evaluation.exceptions import MetricComputationError
from backend.evaluation.metrics.hit_rate import hit_rate_at_k


def test_hit_rate_at_k_is_zero_before_first_relevant_item_appears() -> None:
    retrieved = ["x", "a", "b"]
    relevant = {"a", "b"}

    assert hit_rate_at_k(retrieved, relevant, 1) == 0.0


def test_hit_rate_at_k_is_one_once_a_relevant_item_is_within_cutoff() -> None:
    retrieved = ["x", "a", "b"]
    relevant = {"a", "b"}

    assert hit_rate_at_k(retrieved, relevant, 2) == 1.0


def test_hit_rate_at_k_is_zero_when_nothing_relevant_present() -> None:
    assert hit_rate_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_hit_rate_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(MetricComputationError):
        hit_rate_at_k(["a"], {"a"}, -1)
