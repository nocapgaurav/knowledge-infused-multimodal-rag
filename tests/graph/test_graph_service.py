"""Tests for graph construction orchestration: staleness, validation,
persistence -- using a fake KnowledgeGraphStore and real LocalFilesystemStorage
(against tmp_path). One test runs against real Neo4j to prove the fake and
the real provider produce identical service-level behavior."""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from backend.domain import ChunkModality, PaperId
from backend.graph.exceptions import KnowledgeRepresentationNotFoundError
from backend.graph.interfaces.knowledge_graph_store import KnowledgeGraphStore
from backend.graph.models import GraphSummary, KnowledgeGraph
from backend.graph.providers.neo4j_provider import Neo4jProvider
from backend.graph.repository.graph_repository import GraphRepository
from backend.graph.services.graph_service import GraphService
from backend.storage.local_filesystem import LocalFilesystemStorage

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "kimrag-dev-password"


class _FakeGraphStore(KnowledgeGraphStore):
    """A deterministic, fast stand-in for a real graph database."""

    def __init__(self) -> None:
        self._graphs: dict[PaperId, KnowledgeGraph] = {}
        self.replace_call_count = 0

    def ensure_schema(self) -> None:
        pass

    def replace_graph(self, graph: KnowledgeGraph) -> None:
        self.replace_call_count += 1
        self._graphs[graph.document_id] = graph

    def graph_summary(self, document_id: PaperId) -> GraphSummary:
        graph = self._graphs.get(document_id)
        if graph is None:
            return GraphSummary(node_count=0, relationship_count=0)
        return GraphSummary(
            node_count=graph.node_count, relationship_count=graph.relationship_count
        )


def _seed_knowledge(
    knowledge_storage: LocalFilesystemStorage, document_id: PaperId, text: str = "some text"
) -> str:
    if not knowledge_storage.workspace_exists(document_id):
        knowledge_storage.create_workspace(document_id)
    chunk_id = str(uuid4())
    knowledge_storage.write_json(
        document_id,
        "knowledge_units.json",
        {
            "document_id": str(document_id),
            "count": 1,
            "chunks": [
                {
                    "id": chunk_id,
                    "paper_id": str(document_id),
                    "section_id": None,
                    "order": 0,
                    "modality": ChunkModality.TEXT.value,
                    "text": text,
                    "asset_uri": None,
                    "token_count": None,
                    "source_element_ids": [],
                    "bounding_boxes": [],
                }
            ],
        },
    )
    knowledge_storage.write_json(
        document_id,
        "relationships.json",
        {"document_id": str(document_id), "count": 0, "relationships": []},
    )
    return chunk_id


@pytest.fixture
def storages(tmp_path: Path) -> tuple[LocalFilesystemStorage, LocalFilesystemStorage]:
    knowledge_storage = LocalFilesystemStorage(root=tmp_path / "knowledge")
    graph_storage = LocalFilesystemStorage(root=tmp_path / "graph")
    return knowledge_storage, graph_storage


def _service(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage], store: KnowledgeGraphStore
) -> GraphService:
    knowledge_storage, graph_storage = storages
    repository = GraphRepository(knowledge_storage=knowledge_storage, graph_storage=graph_storage)
    return GraphService(repository=repository, store=store)


def test_normal_build(storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage]) -> None:
    knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id)
    service = _service(storages, _FakeGraphStore())

    result = service.build_graph(document_id)

    assert result.newly_built is True
    assert result.manifest.node_count == 2  # Document + one KnowledgeUnit
    assert result.manifest.relationship_count == 1  # BELONGS_TO


def test_missing_knowledge_representation_raises(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    service = _service(storages, _FakeGraphStore())

    with pytest.raises(KnowledgeRepresentationNotFoundError):
        service.build_graph(PaperId(uuid4()))


def test_second_call_without_force_skips_rebuild(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id)
    store = _FakeGraphStore()
    service = _service(storages, store)

    service.build_graph(document_id)
    calls_after_first = store.replace_call_count
    result = service.build_graph(document_id)

    assert result.newly_built is False
    assert store.replace_call_count == calls_after_first


def test_force_rebuilds_even_when_fresh(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id)
    store = _FakeGraphStore()
    service = _service(storages, store)

    service.build_graph(document_id)
    calls_after_first = store.replace_call_count
    result = service.build_graph(document_id, force=True)

    assert result.newly_built is True
    assert store.replace_call_count > calls_after_first


def test_representation_change_triggers_rebuild(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id, text="version one")
    service = _service(storages, _FakeGraphStore())
    service.build_graph(document_id)

    _seed_knowledge(knowledge_storage, document_id, text="version two")
    result = service.build_graph(document_id)

    assert result.newly_built is True


def test_graph_version_bump_triggers_rebuild_without_representation_change(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    import backend.graph.services.graph_service as graph_service_module

    knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id)
    service = _service(storages, _FakeGraphStore())
    service.build_graph(document_id)

    original_version = graph_service_module.GRAPH_CONSTRUCTION_VERSION
    try:
        graph_service_module.GRAPH_CONSTRUCTION_VERSION = "2.0"
        result = service.build_graph(document_id)
        assert result.newly_built is True
        assert result.manifest.graph_version == "2.0"
    finally:
        graph_service_module.GRAPH_CONSTRUCTION_VERSION = original_version


def test_persistence_writes_graph_manifest(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage], tmp_path: Path
) -> None:
    knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id)
    service = _service(storages, _FakeGraphStore())

    service.build_graph(document_id)

    manifest_payload = json.loads(
        (tmp_path / "graph" / str(document_id) / "graph_manifest.json").read_text()
    )
    assert manifest_payload["document_id"] == str(document_id)
    assert manifest_payload["node_count"] == 2
    assert manifest_payload["relationship_count"] == 1


def test_provider_replacement_produces_identical_business_outcome(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    """Swapping the fake KnowledgeGraphStore for the real Neo4j provider
    changes nothing about the service's own behavior -- same counts, same
    manifest shape."""
    knowledge_storage, _ = storages

    document_id_fake = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id_fake)
    fake_result = _service(storages, _FakeGraphStore()).build_graph(document_id_fake)

    document_id_real = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id_real)
    real_provider = Neo4jProvider(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    try:
        real_result = _service(storages, real_provider).build_graph(document_id_real)

        assert fake_result.manifest.node_count == real_result.manifest.node_count
        assert fake_result.manifest.relationship_count == real_result.manifest.relationship_count
        assert fake_result.newly_built is real_result.newly_built is True
    finally:
        with real_provider._driver.session(database=real_provider._database) as session:
            session.run(
                "MATCH (n:KGNode {paper_id: $paper_id}) DETACH DELETE n",
                paper_id=str(document_id_real),
            )  # test-only cleanup
        real_provider.close()
