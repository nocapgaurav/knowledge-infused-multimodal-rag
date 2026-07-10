"""Tests for Phase 8: citation resolution."""

from backend.generation.citations.citation_resolver import CitationResolver
from backend.generation.models.answer_plan import (
    AnswerPlan,
    AnswerSection,
    ExpectedAnswerType,
    QuestionType,
)
from backend.generation.models.prompt_context import ContextSection, PromptContext


def _plan() -> AnswerPlan:
    return AnswerPlan(
        question_type=QuestionType.FACTUAL,
        expected_answer_type=ExpectedAnswerType.SHORT_FACTUAL,
        required_evidence_groups=1,
        expected_sections=tuple(AnswerSection),
    )


def _prompt(sections) -> PromptContext:
    return PromptContext(
        system_prompt="You are a scientific assistant.",
        grounding_rules=("Cite every claim.",),
        answer_plan=_plan(),
        context_sections=tuple(sections),
        formatting_rules=("Be concise.",),
        user_question="What happened?",
    )


def _section(label="KU1", knowledge_unit_id="abc-123", text="evidence") -> ContextSection:
    return ContextSection(
        citation_label=label, knowledge_unit_id=knowledge_unit_id, text=text, modality="text"
    )


def test_resolves_valid_citation() -> None:
    section = _section(label="KU1", knowledge_unit_id="real-id", text="the evidence text")
    answer = "This is supported [KU1]."

    report = CitationResolver().resolve(answer, _prompt([section]))

    assert len(report.resolved) == 1
    assert report.resolved[0].label == "KU1"
    assert report.resolved[0].knowledge_unit_id == "real-id"
    assert report.resolved[0].text_excerpt == "the evidence text"
    assert report.unresolved == ()


def test_unknown_label_is_unresolved() -> None:
    answer = "This cites something invented [KU99]."

    report = CitationResolver().resolve(answer, _prompt([_section(label="KU1")]))

    assert report.resolved == ()
    assert len(report.unresolved) == 1
    assert report.unresolved[0].label == "KU99"


def test_duplicate_label_usage_resolves_only_once() -> None:
    section = _section(label="KU1")
    answer = "First mention [KU1]. Second mention of the same thing [KU1]."

    report = CitationResolver().resolve(answer, _prompt([section]))

    assert len(report.resolved) == 1


def test_labels_resolved_in_first_appearance_order() -> None:
    sections = [
        _section(label="KU1", knowledge_unit_id="id-1"),
        _section(label="KU2", knowledge_unit_id="id-2"),
    ]
    answer = "Second claim first [KU2]. First claim second [KU1]."

    report = CitationResolver().resolve(answer, _prompt(sections))

    assert [c.label for c in report.resolved] == ["KU2", "KU1"]


def test_no_citations_used_produces_empty_report() -> None:
    answer = "This answer has no citations at all."

    report = CitationResolver().resolve(answer, _prompt([_section()]))

    assert report.resolved == ()
    assert report.unresolved == ()
