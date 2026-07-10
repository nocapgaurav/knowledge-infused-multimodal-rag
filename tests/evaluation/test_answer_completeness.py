"""Unit tests for Answer Completeness (recall-flavored citation metric)."""

from backend.evaluation.metrics.answer_completeness import answer_completeness


def test_answer_completeness_is_fraction_of_expected_that_was_cited() -> None:
    cited = {"a", "x"}
    expected = {"a", "b"}

    result = answer_completeness(cited, expected)

    assert result == 0.5


def test_answer_completeness_is_one_when_everything_expected_was_cited() -> None:
    assert answer_completeness({"a", "b", "x"}, {"a", "b"}) == 1.0


def test_answer_completeness_is_zero_when_nothing_expected_was_cited() -> None:
    assert answer_completeness({"x"}, {"a", "b"}) == 0.0


def test_answer_completeness_is_one_when_expected_citations_are_empty() -> None:
    assert answer_completeness({"x"}, set()) == 1.0
