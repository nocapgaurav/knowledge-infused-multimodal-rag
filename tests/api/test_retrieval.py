"""End-to-end tests for the hybrid evidence retrieval API.

Overrides the retrieval service with a fake -- this test verifies
routing, dependency wiring, and status-code mapping, not the real
retrieval pipeline itself (covered separately in
tests/retrieval/test_retrieval_pipeline_integration.py's real-stack case).
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import get_retrieval_service
from backend.domain import PaperId
from backend.retrieval.exceptions import DocumentNotIndexedError
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
from backend.retrieval.models.retrieval_candidate import DiscoveryMethod


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
            discovery_method=DiscoveryMethod.DENSE_RETRIEVAL,
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


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_retrieval_service] = lambda: _FakeRetrievalService()
    with TestClient(app) as test_client:
        yield test_client


def test_retrieve_evidence_returns_bundle(client: TestClient) -> None:
    document_id = uuid4()

    response = client.post(f"/documents/{document_id}/retrieve", json={"query": "what happened?"})

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == str(document_id)
    assert body["query"] == "what happened?"
    assert len(body["candidates"]) == 1
    assert len(body["evidence_groups"]) == 1
    assert body["manifest"]["retrieval_version"] == "1.0"


def test_retrieve_evidence_returns_404_for_unindexed_document() -> None:
    app = create_app()
    app.dependency_overrides[get_retrieval_service] = lambda: _FakeRetrievalService(
        raise_not_indexed=True
    )
    with TestClient(app) as client:
        response = client.post(f"/documents/{uuid4()}/retrieve", json={"query": "anything"})

    assert response.status_code == 404


def test_retrieve_evidence_rejects_empty_query(client: TestClient) -> None:
    response = client.post(f"/documents/{uuid4()}/retrieve", json={"query": ""})

    assert response.status_code == 422
