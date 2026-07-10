"""Tests for Phase 7: grounding validation."""

import pytest

from backend.generation.exceptions import NoClaimsExtractedError
from backend.generation.grounding.grounding_validator import GroundingValidator
from backend.generation.models.answer_plan import (
    AnswerPlan,
    AnswerSection,
    ExpectedAnswerType,
    QuestionType,
)
from backend.generation.models.grounding_report import ClaimGroundingStatus
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


def test_grounded_claim_with_strong_lexical_overlap() -> None:
    section = _section(
        text="Rayleigh scattering causes shorter wavelengths of light to scatter more strongly."
    )
    answer = (
        "Rayleigh scattering causes shorter wavelengths of light to scatter more strongly [KU1]."
    )

    report = GroundingValidator().validate(answer, _prompt([section]))

    assert len(report.claims) == 1
    assert report.claims[0].status is ClaimGroundingStatus.GROUNDED
    assert report.is_fully_grounded is True


def test_claim_without_citation_is_missing_citation() -> None:
    answer = "The sky is blue."

    report = GroundingValidator().validate(answer, _prompt([_section()]))

    assert report.claims[0].status is ClaimGroundingStatus.MISSING_CITATION
    assert report.is_fully_grounded is False


def test_claim_citing_unknown_label_is_unresolved() -> None:
    answer = "The sky is blue [KU99]."

    report = GroundingValidator().validate(answer, _prompt([_section(label="KU1")]))

    assert report.claims[0].status is ClaimGroundingStatus.UNRESOLVED_CITATION


def test_claim_with_valid_citation_but_no_lexical_overlap_is_unsupported() -> None:
    section = _section(text="The experiment used a sample size of fifty participants.")
    answer = "The results demonstrate a cure for the common cold [KU1]."

    report = GroundingValidator().validate(answer, _prompt([section]))

    assert report.claims[0].status is ClaimGroundingStatus.UNSUPPORTED


def test_multiple_sentences_each_get_their_own_verdict() -> None:
    section = _section(text="The sample included fifty participants across three age groups.")
    answer = (
        "The sample included fifty participants [KU1]. "
        "This is an unrelated invented claim about interstellar travel."
    )

    report = GroundingValidator().validate(answer, _prompt([section]))

    assert len(report.claims) == 2
    assert report.claims[0].status is ClaimGroundingStatus.GROUNDED
    assert report.claims[1].status is ClaimGroundingStatus.MISSING_CITATION


def test_grounded_ratio_reflects_partial_grounding() -> None:
    section = _section(text="The sample included fifty participants across three age groups.")
    answer = "The sample included fifty participants [KU1]. This is unrelated and uncited."

    report = GroundingValidator().validate(answer, _prompt([section]))

    assert report.grounded_ratio == 0.5


def test_empty_answer_raises_no_claims_extracted() -> None:
    with pytest.raises(NoClaimsExtractedError):
        GroundingValidator().validate("   ", _prompt([_section()]))


def test_markdown_numbered_list_markers_are_not_treated_as_claims() -> None:
    """Regression test: a real model response formatted as a numbered
    list ("...0.87.\\n\\n2. **Method Stages**...") previously split into a
    spurious claim consisting only of "2." -- confirmed against real
    Ollama output during end-to-end verification."""
    section = _section(text="The method consists of four stages: ingestion, parsing, chunking.")
    answer = (
        "1. **Performance**: Some finding [KU1].\n\n"
        "2. **Method Stages**: The method consists of four stages [KU1]."
    )

    report = GroundingValidator().validate(answer, _prompt([section]))

    assert all(claim.claim_text not in {"1.", "2."} for claim in report.claims)
    assert all(len(claim.claim_text) > 2 for claim in report.claims)
