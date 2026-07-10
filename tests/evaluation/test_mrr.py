"""Unit tests for reciprocal rank (the basis of Mean Reciprocal Rank)."""

import pytest

from backend.evaluation.metrics.mrr import reciprocal_rank


def test_reciprocal_rank_of_first_relevant_item_at_rank_two() -> None:
    retrieved = ["x", "a", "b"]
    relevant = {"a", "b"}

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)


def test_reciprocal_rank_is_one_when_first_item_is_relevant() -> None:
    retrieved = ["a", "x"]
    relevant = {"a"}

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1.0)


def test_reciprocal_rank_is_zero_when_no_relevant_item_found() -> None:
    retrieved = ["x", "y", "z"]
    relevant = {"a"}

    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_reciprocal_rank_of_empty_retrieved_list_is_zero() -> None:
    assert reciprocal_rank([], {"a"}) == 0.0
