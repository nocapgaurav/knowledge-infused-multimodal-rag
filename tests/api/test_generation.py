"""End-to-end tests for the grounded answer generation API.

Overrides retrieval and generation services with fakes -- this test
verifies routing, dependency wiring, and status-code mapping, not the
real pipeline itself (covered separately in
tests/generation/test_generation_pipeline_integration.py's real-stack case).
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import get_generation_service, get_retrieval_service
from backend.domain import PaperId
from backend.generation.exceptions import GenerationProviderError
from backend.generation.models import (
    AnswerProvenance,
    AnswerStatus,
    GenerationStatistics,
    GenerationTrace,
    GroundedResponse,
    ResolvedCitation,
    SupportingEvidenceItem,
)
from backend.retrieval.exceptions import DocumentNotIndexedError
from backend.retrieval.models import (
    DiscoveryMethod as RetrievalDiscoveryMethod,
)
from backend.retrieval.models import (
    EvidenceBundle,
    EvidenceGroup,
    RankingExplanation,
    RetrievalCandidate,
    RetrievalManifest,
    RetrievalStatistics,
    RetrievalTrace,
    ScoredCandidate,
    SignalScore,
)


class _FakeRetrievalService:
    def __init__(self, *, raise_not_indexed: bool = False) -> None:
        self._raise_not_indexed = raise_not_indexed

    def retrieve(self, document_id: PaperId, query: str) -> EvidenceBundle:
        if self._raise_not_indexed:
            raise DocumentNotIndexedError(document_id=document_id)

        candidate = RetrievalCandidate(
            knowledge_unit_id=uuid4(),
            document_id=document_id,
            section_id=None,
            modality="text",
            text="fake evidence text",
            asset_uri=None,
            reading_order=0,
            citation_count=0,
            dense_similarity=0.8,
            discovery_method=RetrievalDiscoveryMethod.DENSE_RETRIEVAL,
        )
        scored = ScoredCandidate(
            candidate=candidate,
            ranking=RankingExplanation(
                signals=(SignalScore(name="dense_similarity", raw_value=0.8, rank=1),),
                fused_score=0.5,
                final_rank=1,
            ),
        )
        group = EvidenceGroup(
            group_id=str(candidate.knowledge_unit_id),
            primary=scored,
            supporting=(),
            modalities=("text",),
        )
        return EvidenceBundle(
            document_id=document_id,
            query=query,
            candidates=(candidate,),
            evidence_groups=(group,),
            trace=RetrievalTrace(phases=(), dropped=()),
            manifest=RetrievalManifest(
                document_id=document_id,
                query=query,
                retrieval_version="1.0",
                retrieval_strategy_version="1.0",
                representation_version="repr-hash",
                embedding_version="sha-1",
                graph_version="1.0",
                statistics=RetrievalStatistics(
                    candidates_generated=1,
                    candidates_expanded=0,
                    candidates_scored=1,
                    evidence_groups=1,
                    evidence_items=1,
                    duration_ms=5.0,
                ),
                created_at=datetime.now(UTC),
            ),
        )


class _FakeGenerationService:
    def __init__(self, *, raise_provider_error: bool = False) -> None:
        self._raise_provider_error = raise_provider_error

    def generate(self, bundle: EvidenceBundle, config) -> GroundedResponse:  # noqa: ANN001
        if self._raise_provider_error:
            raise GenerationProviderError(reason="simulated failure")

        candidate = bundle.candidates[0]
        return GroundedResponse(
            document_id=bundle.document_id,
            query=bundle.query,
            answer="A fake grounded answer [KU1].",
            executive_summary="A fake grounded answer.",
            supporting_evidence=(
                SupportingEvidenceItem(
                    label="KU1",
                    knowledge_unit_id=str(candidate.knowledge_unit_id),
                    text=candidate.text,
                    modality=candidate.modality,
                ),
            ),
            resolved_citations=(
                ResolvedCitation(
                    label="KU1",
                    knowledge_unit_id=str(candidate.knowledge_unit_id),
                    text_excerpt=candidate.text,
                ),
            ),
            limitations=(),
            references=(f"[KU1] {candidate.text}",),
            warnings=(),
            confidence=0.9,
            answer_status=AnswerStatus.SUFFICIENT_EVIDENCE,
            generation_metadata={"question_type": "factual"},
            prompt_version="1.0",
            model_name="fake-model",
            model_version="fake-version-1",
            generation_trace=GenerationTrace(phases=()),
            generation_statistics=GenerationStatistics(
                context_sections_used=1,
                context_sections_dropped=0,
                claims_total=1,
                claims_grounded=1,
                citations_resolved=1,
                citations_unresolved=0,
                prompt_tokens=10,
                completion_tokens=5,
                duration_ms=5.0,
            ),
            answer_provenance=AnswerProvenance(
                document_id=bundle.document_id,
                retrieval_version=bundle.manifest.retrieval_version,
                retrieval_strategy_version=bundle.manifest.retrieval_strategy_version,
                representation_version=bundle.manifest.representation_version,
                embedding_version=bundle.manifest.embedding_version,
                graph_version=bundle.manifest.graph_version,
                knowledge_unit_ids=(str(candidate.knowledge_unit_id),),
                evidence_bundle_checksum="checksum",
            ),
        )


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_retrieval_service] = lambda: _FakeRetrievalService()
    app.dependency_overrides[get_generation_service] = lambda: _FakeGenerationService()
    with TestClient(app) as test_client:
        yield test_client


def test_generate_answer_returns_grounded_response(client: TestClient) -> None:
    document_id = uuid4()

    response = client.post(f"/documents/{document_id}/generate", json={"query": "why?"})

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == str(document_id)
    assert body["answer_status"] == "sufficient_evidence"
    assert body["confidence"] == 0.9
    assert len(body["resolved_citations"]) == 1


def test_generate_answer_returns_404_when_document_not_indexed() -> None:
    app = create_app()
    app.dependency_overrides[get_retrieval_service] = lambda: _FakeRetrievalService(
        raise_not_indexed=True
    )
    app.dependency_overrides[get_generation_service] = lambda: _FakeGenerationService()
    with TestClient(app) as client:
        response = client.post(f"/documents/{uuid4()}/generate", json={"query": "anything"})

    assert response.status_code == 404


def test_generate_answer_returns_500_on_provider_failure() -> None:
    app = create_app()
    app.dependency_overrides[get_retrieval_service] = lambda: _FakeRetrievalService()
    app.dependency_overrides[get_generation_service] = lambda: _FakeGenerationService(
        raise_provider_error=True
    )
    with TestClient(app) as client:
        response = client.post(f"/documents/{uuid4()}/generate", json={"query": "anything"})

    assert response.status_code == 500


def test_generate_answer_rejects_empty_query(client: TestClient) -> None:
    response = client.post(f"/documents/{uuid4()}/generate", json={"query": ""})

    assert response.status_code == 422
