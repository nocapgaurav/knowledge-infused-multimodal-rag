"""End-to-end tests for the knowledge graph API.

Overrides the graph store with a fake -- this test verifies routing,
dependency wiring, and status-code mapping, not Neo4j itself (covered
separately in tests/graph/test_neo4j_provider.py and
test_graph_service.py's real-Neo4j case).
"""

from collections.abc import Iterator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import get_graph_storage, get_graph_store, get_knowledge_storage
from backend.domain import ChunkModality, PaperId
from backend.graph.interfaces.knowledge_graph_store import KnowledgeGraphStore
from backend.graph.models import GraphSummary, KnowledgeGraph
from backend.storage.local_filesystem import LocalFilesystemStorage


class _FakeGraphStore(KnowledgeGraphStore):
    def __init__(self) -> None:
        self._graphs: dict[PaperId, KnowledgeGraph] = {}

    def ensure_schema(self) -> None:
        pass

    def replace_graph(self, graph: KnowledgeGraph) -> None:
        self._graphs[graph.document_id] = graph

    def graph_summary(self, document_id: PaperId) -> GraphSummary:
        graph = self._graphs.get(document_id)
        if graph is None:
            return GraphSummary(node_count=0, relationship_count=0)
        return GraphSummary(
            node_count=graph.node_count, relationship_count=graph.relationship_count
        )


def _seed_knowledge(knowledge_storage: LocalFilesystemStorage, document_id: PaperId) -> None:
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
                    "text": "some text",
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


@pytest.fixture
def client_and_storage(tmp_path) -> Iterator[tuple[TestClient, LocalFilesystemStorage]]:
    app = create_app()
    knowledge_storage = LocalFilesystemStorage(root=tmp_path / "knowledge")
    graph_storage = LocalFilesystemStorage(root=tmp_path / "graph")
    app.dependency_overrides[get_knowledge_storage] = lambda: knowledge_storage
    app.dependency_overrides[get_graph_storage] = lambda: graph_storage
    app.dependency_overrides[get_graph_store] = lambda: _FakeGraphStore()
    with TestClient(app) as test_client:
        yield test_client, knowledge_storage


def test_build_graph_returns_nodes_relationships_and_status(
    client_and_storage: tuple[TestClient, LocalFilesystemStorage],
) -> None:
    client, knowledge_storage = client_and_storage
    document_id = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id)

    response = client.post(f"/documents/{document_id}/graph")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == str(document_id)
    assert body["nodes"] == 2
    assert body["relationships"] == 1
    assert body["status"] == "GRAPH_CREATED"


def test_build_graph_returns_404_for_document_without_knowledge_representation(
    client_and_storage: tuple[TestClient, LocalFilesystemStorage],
) -> None:
    client, _ = client_and_storage

    response = client.post(f"/documents/{uuid4()}/graph")

    assert response.status_code == 404


def test_second_call_is_idempotent_without_force(
    client_and_storage: tuple[TestClient, LocalFilesystemStorage],
) -> None:
    client, knowledge_storage = client_and_storage
    document_id = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id)

    first = client.post(f"/documents/{document_id}/graph")
    second = client.post(f"/documents/{document_id}/graph")

    assert first.json()["nodes"] == second.json()["nodes"] == 2


def test_force_query_param_rebuilds(
    client_and_storage: tuple[TestClient, LocalFilesystemStorage],
) -> None:
    client, knowledge_storage = client_and_storage
    document_id = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id)

    client.post(f"/documents/{document_id}/graph")
    response = client.post(f"/documents/{document_id}/graph", params={"force": "true"})

    assert response.status_code == 200
    assert response.json()["nodes"] == 2
