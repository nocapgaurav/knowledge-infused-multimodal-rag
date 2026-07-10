"""Unit tests for Evidence Coverage (precision-flavored citation metric)."""

from backend.evaluation.metrics.evidence_coverage import evidence_coverage


def test_evidence_coverage_is_fraction_of_cited_that_was_expected() -> None:
    cited = {"a", "b", "x"}
    expected = {"a", "b"}

    result = evidence_coverage(cited, expected)

    assert result == 2 / 3


def test_evidence_coverage_is_one_when_every_citation_was_expected() -> None:
    assert evidence_coverage({"a"}, {"a", "b"}) == 1.0


def test_evidence_coverage_is_zero_when_nothing_was_cited() -> None:
    assert evidence_coverage(set(), {"a", "b"}) == 0.0


def test_evidence_coverage_is_zero_when_citations_are_all_wrong() -> None:
    assert evidence_coverage({"x", "y"}, {"a", "b"}) == 0.0
