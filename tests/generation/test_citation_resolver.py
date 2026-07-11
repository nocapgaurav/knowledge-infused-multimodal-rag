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


def test_resolves_parenthesized_citation_format() -> None:
    section = _section(label="KU4", knowledge_unit_id="real-id", text="the evidence text")
    answer = "The system combines multiple modalities (KU4)."

    report = CitationResolver().resolve(answer, _prompt([section]))

    assert len(report.resolved) == 1
    assert report.resolved[0].label == "KU4"
    assert report.unresolved == ()


def test_resolves_every_label_in_a_comma_separated_citation_list() -> None:
    sections = [
        _section(label="KU4", knowledge_unit_id="id-4"),
        _section(label="KU8", knowledge_unit_id="id-8"),
    ]
    answer = "The system combines text, tables, and images (KU4, KU8)."

    report = CitationResolver().resolve(answer, _prompt(sections))

    assert [c.label for c in report.resolved] == ["KU4", "KU8"]
    assert report.unresolved == ()


def test_resolves_every_label_in_a_bracketed_citation_list() -> None:
    sections = [
        _section(label="KU1", knowledge_unit_id="id-1"),
        _section(label="KU2", knowledge_unit_id="id-2"),
    ]
    answer = "As shown earlier [KU1, KU2]."

    report = CitationResolver().resolve(answer, _prompt(sections))

    assert [c.label for c in report.resolved] == ["KU1", "KU2"]
    assert report.unresolved == ()


def test_ordinary_parenthesized_text_is_not_a_citation() -> None:
    answer = "As shown before (Doe, 2021) and elsewhere (see Table 1), results hold [KU1]."

    report = CitationResolver().resolve(answer, _prompt([_section(label="KU1")]))

    assert [c.label for c in report.resolved] == ["KU1"]
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


def test_identity_form_citation_resolves_to_its_label() -> None:
    """Regression: real models sometimes cite by the parenthesized identity
    shown next to the label -- observed live with qwen2.5:
    "...provided in (Authors and affiliations (title page))"."""
    section = ContextSection(
        citation_label="KU1",
        knowledge_unit_id="real-id",
        text="Jane Doe University of Somewhere jane@somewhere.edu",
        retrieval_context="Authors and affiliations (title page)",
        modality="text",
    )
    answer = "The affiliations are given in (Authors and affiliations (title page))."

    report = CitationResolver().resolve(answer, _prompt([section]))

    assert [c.label for c in report.resolved] == ["KU1"]


def test_identity_text_not_shown_in_this_prompt_never_resolves() -> None:
    answer = "As shown in (Figure 9), results improved."

    report = CitationResolver().resolve(answer, _prompt([_section(label="KU1")]))

    assert report.resolved == ()
