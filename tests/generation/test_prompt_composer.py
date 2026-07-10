"""Tests for Phase 4: prompt composition."""

from backend.generation.models.answer_plan import (
    AnswerPlan,
    AnswerSection,
    ExpectedAnswerType,
    QuestionType,
)
from backend.generation.models.prompt_context import ContextSection
from backend.generation.prompt.prompt_composer import PromptComposer


def _plan(question_type: QuestionType = QuestionType.FACTUAL) -> AnswerPlan:
    return AnswerPlan(
        question_type=question_type,
        expected_answer_type=ExpectedAnswerType.SHORT_FACTUAL,
        required_evidence_groups=1,
        expected_sections=tuple(AnswerSection),
    )


def _section(label="KU1") -> ContextSection:
    return ContextSection(
        citation_label=label, knowledge_unit_id="abc-123", text="some evidence", modality="text"
    )


def test_compose_produces_structured_not_concatenated_prompt() -> None:
    prompt = PromptComposer().compose("What happened?", _plan(), [_section()])

    assert prompt.system_prompt
    assert prompt.grounding_rules
    assert prompt.formatting_rules
    assert prompt.answer_plan.question_type is QuestionType.FACTUAL
    assert prompt.context_sections[0].citation_label == "KU1"
    assert prompt.user_question == "What happened?"


def test_compose_adds_no_evidence_rule_when_context_is_empty() -> None:
    prompt = PromptComposer().compose("What happened?", _plan(), [])

    assert any("No evidence was found" in rule for rule in prompt.grounding_rules)


def test_compose_does_not_add_no_evidence_rule_when_context_present() -> None:
    prompt = PromptComposer().compose("What happened?", _plan(), [_section()])

    assert not any("No evidence was found" in rule for rule in prompt.grounding_rules)


def test_compose_adds_question_type_specific_formatting_rule() -> None:
    prompt = PromptComposer().compose(
        "Compare A and B.", _plan(QuestionType.COMPARATIVE), [_section()]
    )

    assert any("comparison" in rule.lower() for rule in prompt.formatting_rules)


def test_compose_preserves_all_context_sections_in_order() -> None:
    sections = [_section(label=f"KU{i}") for i in range(1, 4)]

    prompt = PromptComposer().compose("q", _plan(), sections)

    assert [s.citation_label for s in prompt.context_sections] == ["KU1", "KU2", "KU3"]
