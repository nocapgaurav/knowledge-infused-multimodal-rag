"""Unit tests for Grounding Accuracy."""

import pytest

from backend.evaluation.exceptions import MetricComputationError
from backend.evaluation.metrics.grounding_accuracy import grounding_accuracy


def test_grounding_accuracy_is_fraction_of_claims_grounded() -> None:
    assert grounding_accuracy(claims_grounded=3, claims_total=4) == pytest.approx(0.75)


def test_grounding_accuracy_is_one_when_all_claims_grounded() -> None:
    assert grounding_accuracy(claims_grounded=5, claims_total=5) == pytest.approx(1.0)


def test_grounding_accuracy_rejects_non_positive_claims_total() -> None:
    with pytest.raises(MetricComputationError):
        grounding_accuracy(claims_grounded=0, claims_total=0)
