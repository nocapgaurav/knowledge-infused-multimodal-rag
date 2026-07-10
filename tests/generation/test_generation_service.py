"""Tests for full generation orchestration, using a fake provider and real
LocalFilesystemStorage (against tmp_path) -- no Ollama needed here. The
real, no-mock, full-pipeline case is covered separately in
test_generation_pipeline_integration.py.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from backend.generation.citations.citation_resolver import CitationResolver
from backend.generation.context.context_optimizer import ContextOptimizer
from backend.generation.exceptions import GenerationProviderError, MissingEvidenceError
from backend.generation.formatting.response_formatter import ResponseFormatter
from backend.generation.grounding.grounding_validator import GroundingValidator
from backend.generation.interfaces.generation_provider import GenerationProvider
from backend.generation.models.answer_status import AnswerStatus
from backend.generation.models.generation_config import GenerationConfig
from backend.generation.models.prompt_context import PromptContext
from backend.generation.models.provider_response import ProviderResponse
from backend.generation.planner.answer_planner import AnswerPlanner
from backend.generation.prompt.prompt_composer import PromptComposer
from backend.generation.prompt.prompt_validator import PromptValidator
from backend.generation.quality.answer_quality_assessor import AnswerQualityAssessor
from backend.generation.repository.generation_repository import GenerationRepository
from backend.generation.services.generation_service import GenerationService
from backend.generation.validation.generation_validator import GenerationValidator
from backend.retrieval.models import (
    DiscoveryMethod,
    EvidenceGroup,
    RankingExplanation,
    RetrievalCandidate,
    RetrievalManifest,
    RetrievalStatistics,
    RetrievalTrace,
    ScoredCandidate,
    SignalScore,
)
from backend.retrieval.models.evidence_bundle import EvidenceBundle
from backend.storage.local_filesystem import LocalFilesystemStorage


class _FakeProvider(GenerationProvider):
    def __init__(self, response_text: str = "The sky is blue due to scattering [KU1].") -> None:
        self._response_text = response_text
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    def resolve_model_version(self, model: str) -> str:
        return "fake-version-1"

    def generate(self, prompt_context: PromptContext, config: GenerationConfig) -> ProviderResponse:
        self.calls += 1
        return ProviderResponse(
            text=self._response_text, prompt_tokens=50, completion_tokens=10, duration_ms=5.0
        )


class _FailingProvider(GenerationProvider):
    @property
    def provider_name(self) -> str:
        return "fake"

    def resolve_model_version(self, model: str) -> str:
        return "fake-version-1"

    def generate(self, prompt_context: PromptContext, config: GenerationConfig) -> ProviderResponse:
        raise GenerationProviderError(reason="simulated provider failure")


def _candidate(document_id, text="Rayleigh scattering causes the sky to appear blue."):
    return RetrievalCandidate(
        knowledge_unit_id=uuid4(),
        document_id=document_id,
        section_id=None,
        modality="text",
        text=text,
        asset_uri=None,
        reading_order=0,
        citation_count=0,
        dense_similarity=0.9,
        discovery_method=DiscoveryMethod.DENSE_RETRIEVAL,
    )


def _bundle(document_id, candidates) -> EvidenceBundle:
    groups = []
    for i, candidate in enumerate(candidates, start=1):
        scored = ScoredCandidate(
            candidate=candidate,
            ranking=RankingExplanation(
                signals=(SignalScore(name="dense_similarity", raw_value=0.9, rank=i),),
                fused_score=1.0 / i,
                final_rank=i,
            ),
        )
        groups.append(
            EvidenceGroup(
                group_id=str(candidate.knowledge_unit_id),
                primary=scored,
                supporting=(),
                modalities=("text",),
            )
        )
    return EvidenceBundle(
        document_id=document_id,
        query="Why is the sky blue?",
        candidates=tuple(candidates),
        evidence_groups=tuple(groups),
        trace=RetrievalTrace(phases=(), dropped=()),
        manifest=RetrievalManifest(
            document_id=document_id,
            query="Why is the sky blue?",
            retrieval_version="1.0",
            retrieval_strategy_version="1.0",
            representation_version="repr-hash",
            embedding_version="embed-hash",
            graph_version="1.0",
            statistics=RetrievalStatistics(
                candidates_generated=len(candidates),
                candidates_expanded=0,
                candidates_scored=len(candidates),
                evidence_groups=len(groups),
                evidence_items=len(candidates),
                duration_ms=1.0,
            ),
            created_at=datetime.now(UTC),
        ),
    )


def _config() -> GenerationConfig:
    return GenerationConfig(
        provider="fake",
        model="fake-model",
        temperature=0.1,
        top_p=0.9,
        max_tokens=300,
        context_window=4096,
    )


@pytest.fixture
def generation_storage(tmp_path: Path) -> LocalFilesystemStorage:
    return LocalFilesystemStorage(root=tmp_path / "generation")


def _service(
    generation_storage: LocalFilesystemStorage, provider: GenerationProvider
) -> GenerationService:
    return GenerationService(
        repository=GenerationRepository(generation_storage=generation_storage),
        provider=provider,
        planner=AnswerPlanner(),
        context_optimizer=ContextOptimizer(),
        prompt_composer=PromptComposer(),
        prompt_validator=PromptValidator(),
        grounding_validator=GroundingValidator(),
        citation_resolver=CitationResolver(),
        quality_assessor=AnswerQualityAssessor(),
        response_formatter=ResponseFormatter(),
        generation_validator=GenerationValidator(),
    )


def test_generate_produces_a_grounded_response(generation_storage: LocalFilesystemStorage) -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [_candidate(document_id)])
    service = _service(generation_storage, _FakeProvider())

    response = service.generate(bundle, _config())

    assert response.document_id == document_id
    assert response.query == bundle.query
    assert "KU1" in response.answer
    assert len(response.resolved_citations) == 1
    assert response.answer_status is AnswerStatus.SUFFICIENT_EVIDENCE
    assert response.model_name == "fake-model"
    assert response.model_version == "fake-version-1"


def test_generate_traces_all_nine_phases(generation_storage: LocalFilesystemStorage) -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [_candidate(document_id)])
    service = _service(generation_storage, _FakeProvider())

    response = service.generate(bundle, _config())

    traced = [phase.phase for phase in response.generation_trace.phases]
    assert traced == [
        "answer_planning",
        "context_optimization",
        "prompt_composition",
        "prompt_validation",
        "generation",
        "grounding_validation",
        "citation_resolution",
        "answer_quality_assessment",
        "response_formatting",
    ]


def test_generate_persists_a_generation_manifest(
    generation_storage: LocalFilesystemStorage, tmp_path: Path
) -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [_candidate(document_id)])
    service = _service(generation_storage, _FakeProvider())

    service.generate(bundle, _config())

    manifest_path = tmp_path / "generation" / str(document_id) / "generation_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["document_id"] == str(document_id)
    assert payload["provider"] == "fake"


def test_generate_with_no_evidence_raises_missing_evidence(
    generation_storage: LocalFilesystemStorage,
) -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [])
    service = _service(generation_storage, _FakeProvider())

    with pytest.raises(MissingEvidenceError):
        service.generate(bundle, _config())


def test_provider_failure_propagates(generation_storage: LocalFilesystemStorage) -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [_candidate(document_id)])
    service = _service(generation_storage, _FailingProvider())

    with pytest.raises(GenerationProviderError):
        service.generate(bundle, _config())


def test_unsupported_answer_lowers_status_and_populates_limitations(
    generation_storage: LocalFilesystemStorage,
) -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [_candidate(document_id, text="Completely unrelated content.")])
    provider = _FakeProvider(response_text="A totally unsupported invented claim [KU1].")
    service = _service(generation_storage, provider)

    response = service.generate(bundle, _config())

    assert response.answer_status is not AnswerStatus.SUFFICIENT_EVIDENCE
    assert response.limitations


def test_provider_replacement_requires_no_service_changes(
    generation_storage: LocalFilesystemStorage,
) -> None:
    """The same GenerationService runs unmodified against a fake provider
    here and against the real OllamaProvider in
    test_generation_pipeline_integration.py -- proving generation business
    logic never needed to change to support a different backend.
    """
    document_id = uuid4()
    bundle = _bundle(document_id, [_candidate(document_id)])
    service = _service(generation_storage, _FakeProvider())

    response = service.generate(bundle, _config())

    assert response.generation_statistics.claims_total >= 1
