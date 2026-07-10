"""Unit tests for Citation Accuracy."""

import pytest

from backend.evaluation.metrics.citation_accuracy import citation_accuracy


def test_citation_accuracy_is_fraction_resolved() -> None:
    result = citation_accuracy(citations_resolved=3, citations_unresolved=1)

    assert result == pytest.approx(0.75)


def test_citation_accuracy_is_vacuously_one_when_nothing_attempted() -> None:
    assert citation_accuracy(citations_resolved=0, citations_unresolved=0) == 1.0


def test_citation_accuracy_is_zero_when_nothing_resolved() -> None:
    assert citation_accuracy(citations_resolved=0, citations_unresolved=2) == 0.0
