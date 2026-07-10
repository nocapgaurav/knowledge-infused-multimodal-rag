"""Unit tests for Unsupported Claim Rate."""

import pytest

from backend.evaluation.exceptions import MetricComputationError
from backend.evaluation.metrics.grounding_accuracy import grounding_accuracy
from backend.evaluation.metrics.unsupported_claim_rate import unsupported_claim_rate


def test_unsupported_claim_rate_is_fraction_of_claims_that_failed_grounding() -> None:
    assert unsupported_claim_rate(claims_grounded=3, claims_total=4) == pytest.approx(0.25)


def test_unsupported_claim_rate_is_zero_when_all_claims_grounded() -> None:
    assert unsupported_claim_rate(claims_grounded=5, claims_total=5) == 0.0


def test_unsupported_claim_rate_is_complement_of_grounding_accuracy() -> None:
    grounded, total = 3, 7

    assert unsupported_claim_rate(grounded, total) == pytest.approx(
        1 - grounding_accuracy(grounded, total)
    )


def test_unsupported_claim_rate_rejects_non_positive_claims_total() -> None:
    with pytest.raises(MetricComputationError):
        unsupported_claim_rate(claims_grounded=0, claims_total=0)
