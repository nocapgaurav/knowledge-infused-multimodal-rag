"""Tests for full retrieval orchestration, using fakes for the two
providers and real LocalFilesystemStorage (against tmp_path) for
manifests -- no Docker needed here. The real, no-mock, full-pipeline case
is covered separately in test_retrieval_pipeline_integration.py.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from backend.domain import PaperId
from backend.embeddings.interfaces.embedding_provider import EmbeddingProvider
from backend.retrieval.assembly.evidence_assembler import AssemblyBudget, EvidenceAssembler
from backend.retrieval.candidate.candidate_generator import CandidateGenerator
from backend.retrieval.evaluation.evidence_evaluator import EvidenceEvaluator
from backend.retrieval.expansion.graph_expander import ExpansionBudget, GraphExpander
from backend.retrieval.interfaces.graph_retriever import GraphRetriever
from backend.retrieval.interfaces.vector_retriever import VectorRetriever
from backend.retrieval.models import GraphNeighbor, TraversalDirection
from backend.retrieval.repository.retrieval_repository import RetrievalRepository
from backend.retrieval.services.retrieval_service import RetrievalService
from backend.retrieval.validation.retrieval_validator import RetrievalValidator
from backend.search.models import EqualityFilter, SearchResult, VectorPoint
from backend.storage.local_filesystem import LocalFilesystemStorage


class _FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def model_version(self) -> str:
        return "fake-1"

    @property
    def embedding_dimension(self) -> int:
        return 4

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeVectorRetriever(VectorRetriever):
    def __init__(self, seed: VectorPoint, hydratable: dict[UUID, VectorPoint]) -> None:
        self._seed = seed
        self._hydratable = hydratable

    def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        limit: int,
        filters: Sequence[EqualityFilter] = (),
    ) -> list[SearchResult]:
        return [SearchResult(id=self._seed.id, score=0.95, payload=dict(self._seed.payload))]

    def retrieve_by_ids(self, collection: str, ids: Sequence[UUID]) -> list[VectorPoint]:
        return [self._hydratable[i] for i in ids if i in self._hydratable]


class _FakeGraphRetriever(GraphRetriever):
    def __init__(self, edges: dict[tuple[str, str, str], list[str]]) -> None:
        self._edges = edges  # (node_id, relationship_type, direction.value) -> [neighbor_id, ...]
        self._labels: dict[str, tuple[str, ...]] = {}

    def register_labels(self, node_id: str, labels: tuple[str, ...]) -> None:
        self._labels[node_id] = labels

    def neighbors(
        self,
        node_ids: Sequence[str],
        relationship_types: Sequence[str],
        direction: TraversalDirection,
    ) -> list[GraphNeighbor]:
        found: list[GraphNeighbor] = []
        for node_id in node_ids:
            for relationship_type in relationship_types:
                key = (node_id, relationship_type, direction.value)
                for neighbor_id in self._edges.get(key, []):
                    found.append(
                        GraphNeighbor(
                            source_id=node_id,
                            neighbor_id=neighbor_id,
                            neighbor_labels=self._labels.get(neighbor_id, ("KnowledgeUnit",)),
                            relationship_type=relationship_type,
                            direction=direction,
                        )
                    )
        return found


def _point(document_id: PaperId, **payload: object) -> VectorPoint:
    point_id = uuid4()
    defaults = {
        "knowledge_unit_id": str(point_id),
        "document_id": str(document_id),
        "section_id": None,
        "modality": "text",
        "text": "text",
        "asset_uri": None,
        "reading_order": 0,
        "citation_count": 0,
    }
    defaults.update(payload)
    return VectorPoint(id=point_id, vector=[0.1, 0.2, 0.3, 0.4], payload=defaults)


@pytest.fixture
def repository(tmp_path: Path) -> RetrievalRepository:
    embeddings_storage = LocalFilesystemStorage(root=tmp_path / "embeddings")
    index_storage = LocalFilesystemStorage(root=tmp_path / "index")
    graph_storage = LocalFilesystemStorage(root=tmp_path / "graph")
    retrieval_storage = LocalFilesystemStorage(root=tmp_path / "retrieval")
    return RetrievalRepository(
        embeddings_storage=embeddings_storage,
        index_storage=index_storage,
        graph_storage=graph_storage,
        retrieval_storage=retrieval_storage,
    )


def _seed_manifests(repository: RetrievalRepository, document_id: PaperId, collection: str) -> None:
    now = datetime.now(UTC).isoformat()
    repository._embeddings_storage.create_workspace(document_id)
    repository._embeddings_storage.write_json(
        document_id,
        "manifest.json",
        {
            "document_id": str(document_id),
            "model_name": "BAAI/bge-m3",
            "model_version": "sha-1",
            "embedding_dimension": 4,
            "artifact_version": "1.0",
            "source_representation_version": "repr-hash",
            "embedding_count": 3,
            "failed_count": 0,
            "skipped_image_count": 0,
            "created_at": now,
        },
    )
    repository._index_storage.create_workspace(document_id)
    repository._index_storage.write_json(
        document_id,
        "index_manifest.json",
        {
            "document_id": str(document_id),
            "collection_name": collection,
            "vector_dimension": 4,
            "distance_metric": "cosine",
            "embedding_model": "BAAI/bge-m3",
            "embedding_version": "sha-1",
            "artifact_version": "1.0",
            "source_embedding_manifest": "manifest-hash",
            "checksum": "checksum-1",
            "indexed_vectors": 3,
            "failed_vectors": 0,
            "created_at": now,
        },
    )
    repository._graph_storage.create_workspace(document_id)
    repository._graph_storage.write_json(
        document_id,
        "graph_manifest.json",
        {
            "document_id": str(document_id),
            "artifact_version": "1.0",
            "graph_version": "1.0",
            "node_count": 4,
            "relationship_count": 3,
            "checksum": "graph-checksum",
            "source_representation_version": "repr-hash",
            "created_at": now,
        },
    )


def _build_service(
    repository: RetrievalRepository,
    vector_retriever: VectorRetriever,
    graph_retriever: GraphRetriever,
) -> RetrievalService:
    return RetrievalService(
        repository=repository,
        candidate_generator=CandidateGenerator(
            _FakeEmbeddingProvider(), vector_retriever, top_k=10
        ),
        graph_expander=GraphExpander(graph_retriever, vector_retriever),
        evaluator=EvidenceEvaluator(),
        assembler=EvidenceAssembler(),
        validator=RetrievalValidator(),
        expansion_budget=ExpansionBudget(),
        assembly_budget=AssemblyBudget(),
    )


def test_retrieve_produces_a_bundle_with_expanded_evidence(
    repository: RetrievalRepository,
) -> None:
    document_id = PaperId(uuid4())
    collection = "fake-collection"
    _seed_manifests(repository, document_id, collection)

    seed_point = _point(document_id, text="seed paragraph")
    next_point = _point(document_id, text="next paragraph")
    figure_point = _point(document_id, text="referenced figure", modality="figure")

    vector_retriever = _FakeVectorRetriever(
        seed=seed_point,
        hydratable={next_point.id: next_point, figure_point.id: figure_point},
    )
    graph_retriever = _FakeGraphRetriever(
        edges={
            (str(seed_point.id), "NEXT", "outgoing"): [str(next_point.id)],
            (str(next_point.id), "REFERENCES", "outgoing"): [str(figure_point.id)],
        }
    )

    service = _build_service(repository, vector_retriever, graph_retriever)
    bundle = service.retrieve(document_id, "what does the paper show?")

    assert bundle.document_id == document_id
    assert len(bundle.candidates) == 3
    assert bundle.evidence_groups
    assert bundle.manifest.embedding_version == "sha-1"
    assert bundle.manifest.graph_version == "1.0"
    assert bundle.manifest.representation_version == "repr-hash"
    assert bundle.trace.phases[0].phase == "candidate_generation"


def test_retrieve_persists_a_retrieval_manifest(
    repository: RetrievalRepository, tmp_path: Path
) -> None:
    document_id = PaperId(uuid4())
    collection = "fake-collection"
    _seed_manifests(repository, document_id, collection)
    seed_point = _point(document_id)
    vector_retriever = _FakeVectorRetriever(seed=seed_point, hydratable={})
    graph_retriever = _FakeGraphRetriever(edges={})

    service = _build_service(repository, vector_retriever, graph_retriever)
    service.retrieve(document_id, "a question")

    saved_path = tmp_path / "retrieval" / str(document_id) / "retrieval_manifest.json"
    assert saved_path.exists()


def test_retrieval_is_deterministic_across_repeated_calls(
    repository: RetrievalRepository,
) -> None:
    document_id = PaperId(uuid4())
    collection = "fake-collection"
    _seed_manifests(repository, document_id, collection)
    seed_point = _point(document_id)
    next_point = _point(document_id, text="next")
    vector_retriever = _FakeVectorRetriever(seed=seed_point, hydratable={next_point.id: next_point})
    graph_retriever = _FakeGraphRetriever(
        edges={(str(seed_point.id), "NEXT", "outgoing"): [str(next_point.id)]}
    )
    service = _build_service(repository, vector_retriever, graph_retriever)

    first = service.retrieve(document_id, "same question")
    second = service.retrieve(document_id, "same question")

    assert [c.knowledge_unit_id for c in first.candidates] == [
        c.knowledge_unit_id for c in second.candidates
    ]
    assert [g.primary.ranking.fused_score for g in first.evidence_groups] == [
        g.primary.ranking.fused_score for g in second.evidence_groups
    ]


def test_provider_replacement_requires_no_service_changes(
    repository: RetrievalRepository,
) -> None:
    """The same RetrievalService, CandidateGenerator, and GraphExpander
    classes run unmodified against a fake VectorRetriever/GraphRetriever
    here and against real QdrantRetriever/Neo4jRetriever in
    test_retrieval_pipeline_integration.py -- proving retrieval business
    logic never needed to change to support a different backend.
    """
    document_id = PaperId(uuid4())
    collection = "fake-collection"
    _seed_manifests(repository, document_id, collection)
    seed_point = _point(document_id)
    vector_retriever = _FakeVectorRetriever(seed=seed_point, hydratable={})
    graph_retriever = _FakeGraphRetriever(edges={})

    service = _build_service(repository, vector_retriever, graph_retriever)
    bundle = service.retrieve(document_id, "a question")

    assert bundle.manifest.statistics.candidates_generated == 1
