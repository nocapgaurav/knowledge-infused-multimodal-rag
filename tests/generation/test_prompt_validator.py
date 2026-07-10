"""Tests for Phase 5: prompt validation."""

import pytest

from backend.generation.exceptions import (
    CitationPlaceholderError,
    ContextIntegrityError,
    DuplicatePromptEvidenceError,
    MissingEvidenceError,
    TokenBudgetExceededError,
)
from backend.generation.models.answer_plan import (
    AnswerPlan,
    AnswerSection,
    ExpectedAnswerType,
    QuestionType,
)
from backend.generation.models.generation_config import GenerationConfig
from backend.generation.models.prompt_context import ContextSection, PromptContext
from backend.generation.prompt.prompt_validator import PromptValidator


def _plan(requires_citations: bool = True) -> AnswerPlan:
    return AnswerPlan(
        question_type=QuestionType.FACTUAL,
        expected_answer_type=ExpectedAnswerType.SHORT_FACTUAL,
        required_evidence_groups=1,
        requires_citations=requires_citations,
        expected_sections=tuple(AnswerSection),
    )


def _section(label="KU1", knowledge_unit_id="abc-123", text="evidence") -> ContextSection:
    return ContextSection(
        citation_label=label, knowledge_unit_id=knowledge_unit_id, text=text, modality="text"
    )


def _prompt(sections=(), plan=None) -> PromptContext:
    return PromptContext(
        system_prompt="You are a scientific assistant.",
        grounding_rules=("Cite every claim.",),
        answer_plan=plan or _plan(),
        context_sections=tuple(sections),
        formatting_rules=("Be concise.",),
        user_question="What happened?",
    )


def _config(context_window=4096, max_tokens=800) -> GenerationConfig:
    return GenerationConfig(
        provider="ollama",
        model="qwen2.5:7b-instruct",
        temperature=0.1,
        top_p=0.9,
        max_tokens=max_tokens,
        context_window=context_window,
    )


def test_valid_prompt_passes() -> None:
    PromptValidator().validate(_prompt([_section()]), _config())  # should not raise


def test_token_budget_exceeded_raises() -> None:
    prompt = _prompt([_section(text="x" * 10_000)])

    with pytest.raises(TokenBudgetExceededError):
        PromptValidator().validate(prompt, _config(context_window=100, max_tokens=10))


def test_missing_evidence_raises_when_citations_required() -> None:
    prompt = _prompt([], plan=_plan(requires_citations=True))

    with pytest.raises(MissingEvidenceError):
        PromptValidator().validate(prompt, _config())


def test_missing_evidence_does_not_raise_when_citations_not_required() -> None:
    prompt = _prompt([], plan=_plan(requires_citations=False))

    PromptValidator().validate(prompt, _config())  # should not raise


def test_duplicate_knowledge_unit_id_raises() -> None:
    prompt = _prompt(
        [
            _section(label="KU1", knowledge_unit_id="same-id"),
            _section(label="KU2", knowledge_unit_id="same-id"),
        ]
    )

    with pytest.raises(DuplicatePromptEvidenceError):
        PromptValidator().validate(prompt, _config())


def test_malformed_citation_label_raises() -> None:
    prompt = _prompt([_section(label="not-a-label")])

    with pytest.raises(CitationPlaceholderError):
        PromptValidator().validate(prompt, _config())


def test_duplicate_citation_label_raises() -> None:
    prompt = _prompt(
        [
            _section(label="KU1", knowledge_unit_id="id-1"),
            _section(label="KU1", knowledge_unit_id="id-2"),
        ]
    )

    with pytest.raises(CitationPlaceholderError):
        PromptValidator().validate(prompt, _config())


def test_blank_grounding_rule_raises_context_integrity_error() -> None:
    prompt = PromptContext(
        system_prompt="You are a scientific assistant.",
        grounding_rules=("   ",),
        answer_plan=_plan(),
        context_sections=(_section(),),
        formatting_rules=("Be concise.",),
        user_question="What happened?",
    )

    with pytest.raises(ContextIntegrityError):
        PromptValidator().validate(prompt, _config())
