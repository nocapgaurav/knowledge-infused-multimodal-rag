"""Integration tests for the real Ollama provider.

Runs against a real, locally running Ollama instance with
`qwen2.5:7b-instruct` pulled (see the final report for why this model was
chosen). Not a fake -- these tests make real generation calls.
"""

import pytest

from backend.generation.exceptions import GenerationProviderError
from backend.generation.models.answer_plan import (
    AnswerPlan,
    AnswerSection,
    ExpectedAnswerType,
    QuestionType,
)
from backend.generation.models.generation_config import GenerationConfig
from backend.generation.models.prompt_context import ContextSection, PromptContext
from backend.generation.providers.ollama_provider import OllamaProvider

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen2.5:7b-instruct"


@pytest.fixture
def provider() -> OllamaProvider:
    return OllamaProvider(host=OLLAMA_HOST)


def _config(model: str = MODEL) -> GenerationConfig:
    return GenerationConfig(
        provider="ollama",
        model=model,
        temperature=0.0,
        top_p=0.9,
        max_tokens=60,
        context_window=2048,
    )


def _prompt() -> PromptContext:
    plan = AnswerPlan(
        question_type=QuestionType.FACTUAL,
        expected_answer_type=ExpectedAnswerType.SHORT_FACTUAL,
        required_evidence_groups=1,
        expected_sections=tuple(AnswerSection),
    )
    section = ContextSection(
        citation_label="KU1",
        knowledge_unit_id="abc-123",
        text="Rayleigh scattering causes the sky to appear blue.",
        modality="text",
    )
    return PromptContext(
        system_prompt="You are a scientific assistant. Answer only using the evidence given.",
        grounding_rules=("Cite every claim using the given labels like [KU1].",),
        answer_plan=plan,
        context_sections=(section,),
        formatting_rules=("Answer in one short sentence.",),
        user_question="Why is the sky blue?",
    )


def test_provider_name_is_ollama(provider: OllamaProvider) -> None:
    assert provider.provider_name == "ollama"


def test_resolve_model_version_returns_a_real_digest(provider: OllamaProvider) -> None:
    version = provider.resolve_model_version(MODEL)

    assert isinstance(version, str)
    assert len(version) > 10


def test_resolve_model_version_raises_for_unknown_model(provider: OllamaProvider) -> None:
    with pytest.raises(GenerationProviderError):
        provider.resolve_model_version("does-not-exist:latest")


def test_generate_returns_real_text_with_token_accounting(provider: OllamaProvider) -> None:
    result = provider.generate(_prompt(), _config())

    assert result.text.strip() != ""
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.duration_ms > 0


def test_generate_uses_the_citation_label_from_the_prompt(provider: OllamaProvider) -> None:
    result = provider.generate(_prompt(), _config())

    assert "KU1" in result.text


def test_generate_raises_for_unknown_model(provider: OllamaProvider) -> None:
    with pytest.raises(GenerationProviderError):
        provider.generate(_prompt(), _config(model="does-not-exist:latest"))


def test_generate_is_deterministic_at_zero_temperature(provider: OllamaProvider) -> None:
    first = provider.generate(_prompt(), _config())
    second = provider.generate(_prompt(), _config())

    assert first.text == second.text


def test_render_section_includes_structural_identity() -> None:
    from backend.generation.models.prompt_context import ContextSection
    from backend.generation.providers.ollama_provider import _render_section

    identified = ContextSection(
        citation_label="KU1",
        knowledge_unit_id="abc",
        text="Knowledge-Infused Multimodal QA",
        retrieval_context="Title of this paper",
        modality="text",
    )
    plain = ContextSection(
        citation_label="KU2", knowledge_unit_id="def", text="Body text.", modality="text"
    )

    assert _render_section(identified) == (
        "[KU1] (Title of this paper) Knowledge-Infused Multimodal QA"
    )
    assert _render_section(plain) == "[KU2] Body text."
