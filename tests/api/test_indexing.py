"""End-to-end tests for the search index API.

Overrides the vector store with a fake -- this test verifies routing,
dependency wiring, and status-code mapping, not Qdrant itself (covered
separately in tests/search/test_qdrant_provider.py and
test_indexing_service.py's real-Qdrant case).
"""

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import get_embeddings_storage, get_knowledge_storage, get_vector_store
from backend.domain import ChunkModality, PaperId
from backend.search.interfaces.vector_store import VectorStore
from backend.search.models import (
    CollectionInfo,
    DistanceMetric,
    EqualityFilter,
    SearchResult,
    VectorPoint,
)
from backend.storage.local_filesystem import LocalFilesystemStorage


class _FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self._collections: dict[str, tuple[int, DistanceMetric]] = {}
        self._points: dict[str, dict[UUID, VectorPoint]] = {}

    def collection_exists(self, collection: str) -> bool:
        return collection in self._collections

    def ensure_collection(self, collection, dimension, distance, indexed_fields) -> None:
        if collection not in self._collections:
            self._collections[collection] = (dimension, distance)
            self._points[collection] = {}

    def collection_info(self, collection: str) -> CollectionInfo | None:
        if collection not in self._collections:
            return None
        dimension, distance = self._collections[collection]
        return CollectionInfo(
            name=collection,
            dimension=dimension,
            distance=distance,
            point_count=len(self._points[collection]),
        )

    def upsert(self, collection: str, points: Sequence[VectorPoint]) -> None:
        for point in points:
            self._points[collection][point.id] = point

    def count(self, collection: str, filters: Sequence[EqualityFilter] = ()) -> int:
        points = self._points.get(collection, {}).values()
        return sum(1 for p in points if all(p.payload.get(f.field) == f.value for f in filters))

    def retrieve(self, collection: str, point_ids: Sequence[UUID]) -> list[VectorPoint]:
        points = self._points.get(collection, {})
        return [points[pid] for pid in point_ids if pid in points]

    def search(
        self, collection, query_vector, limit, filters: Sequence[EqualityFilter] = ()
    ) -> list[SearchResult]:
        return []


def _seed_knowledge(knowledge_storage: LocalFilesystemStorage, document_id: PaperId) -> str:
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
    return chunk_id


def _seed_embeddings(
    embeddings_storage: LocalFilesystemStorage, document_id: PaperId, chunk_id: str
) -> None:
    embeddings_storage.create_workspace(document_id)
    embeddings_storage.write_json(
        document_id,
        "embeddings.json",
        {
            "document_id": str(document_id),
            "count": 1,
            "embeddings": [
                {
                    "embedding_id": str(uuid4()),
                    "knowledge_unit_id": chunk_id,
                    "paper_id": str(document_id),
                    "target": "text",
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "model_name": "BAAI/bge-m3",
                    "model_version": "sha-1",
                    "embedding_dimension": 4,
                    "checksum": "abc",
                    "artifact_version": "1.0",
                    "source_representation_version": "repr-hash",
                    "created_at": datetime.now(UTC).isoformat(),
                }
            ],
        },
    )
    embeddings_storage.write_json(
        document_id,
        "manifest.json",
        {
            "document_id": str(document_id),
            "model_name": "BAAI/bge-m3",
            "model_version": "sha-1",
            "embedding_dimension": 4,
            "artifact_version": "1.0",
            "source_representation_version": "repr-hash",
            "embedding_count": 1,
            "failed_count": 0,
            "skipped_image_count": 0,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )


@pytest.fixture
def client_and_storages(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, LocalFilesystemStorage, LocalFilesystemStorage]]:
    app = create_app()
    knowledge_storage = LocalFilesystemStorage(root=tmp_path / "knowledge")
    embeddings_storage = LocalFilesystemStorage(root=tmp_path / "embeddings")
    app.dependency_overrides[get_knowledge_storage] = lambda: knowledge_storage
    app.dependency_overrides[get_embeddings_storage] = lambda: embeddings_storage
    app.dependency_overrides[get_vector_store] = lambda: _FakeVectorStore()
    with TestClient(app) as test_client:
        yield test_client, knowledge_storage, embeddings_storage


def test_index_document_returns_collection_and_count(
    client_and_storages: tuple[TestClient, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    client, knowledge_storage, embeddings_storage = client_and_storages
    document_id = PaperId(uuid4())
    chunk_id = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_id)

    response = client.post(f"/documents/{document_id}/index")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == str(document_id)
    assert body["indexed_vectors"] == 1
    assert body["status"] == "INDEXED"
    assert "kimrag" in body["collection"]


def test_index_document_returns_404_for_unembedded_document(
    client_and_storages: tuple[TestClient, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    client, _, _ = client_and_storages

    response = client.post(f"/documents/{uuid4()}/index")

    assert response.status_code == 404


def test_second_call_is_idempotent_without_force(
    client_and_storages: tuple[TestClient, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    client, knowledge_storage, embeddings_storage = client_and_storages
    document_id = PaperId(uuid4())
    chunk_id = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_id)

    first = client.post(f"/documents/{document_id}/index")
    second = client.post(f"/documents/{document_id}/index")

    assert first.json()["indexed_vectors"] == 1
    assert second.json()["indexed_vectors"] == 1


def test_force_query_param_reindexes(
    client_and_storages: tuple[TestClient, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    client, knowledge_storage, embeddings_storage = client_and_storages
    document_id = PaperId(uuid4())
    chunk_id = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_id)

    client.post(f"/documents/{document_id}/index")
    response = client.post(f"/documents/{document_id}/index", params={"force": "true"})

    assert response.status_code == 200
    assert response.json()["indexed_vectors"] == 1
