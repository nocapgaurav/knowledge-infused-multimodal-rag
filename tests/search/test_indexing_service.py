"""Tests for indexing orchestration: staleness, retry, partial success,
persistence -- using a fake VectorStore and real LocalFilesystemStorage
(against tmp_path). One test runs against real Qdrant to prove the fake
and the real provider produce identical service-level behavior."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from backend.domain import ChunkModality, PaperId
from backend.search.exceptions import (
    EmbeddingArtifactsNotFoundError,
    MultiCollectionIndexingNotSupportedError,
    NoVectorsIndexedError,
)
from backend.search.interfaces.vector_store import VectorStore
from backend.search.models import (
    CollectionInfo,
    DistanceMetric,
    EqualityFilter,
    SearchResult,
    VectorPoint,
)
from backend.search.providers.qdrant_provider import QdrantProvider
from backend.search.repository.index_repository import IndexRepository
from backend.search.services.indexing_service import IndexingService
from backend.storage.local_filesystem import LocalFilesystemStorage


class _FakeVectorStore(VectorStore):
    """A deterministic, fast stand-in for a real vector database."""

    def __init__(self, fail_first_n_upserts: int = 0) -> None:
        self._collections: dict[str, tuple[int, DistanceMetric]] = {}
        self._points: dict[str, dict[UUID, VectorPoint]] = {}
        self._fail_first_n_upserts = fail_first_n_upserts
        self.upsert_call_count = 0

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
        from backend.search.exceptions import VectorStoreError

        self.upsert_call_count += 1
        if self.upsert_call_count <= self._fail_first_n_upserts:
            raise VectorStoreError(reason="simulated transient failure")
        for point in points:
            self._points[collection][point.id] = point

    def count(self, collection: str, filters: Sequence[EqualityFilter] = ()) -> int:
        return sum(1 for p in self._points.get(collection, {}).values() if _matches(p, filters))

    def retrieve(self, collection: str, point_ids: Sequence[UUID]) -> list[VectorPoint]:
        points = self._points.get(collection, {})
        return [points[pid] for pid in point_ids if pid in points]

    def search(
        self, collection, query_vector, limit, filters: Sequence[EqualityFilter] = ()
    ) -> list[SearchResult]:
        matches = [p for p in self._points.get(collection, {}).values() if _matches(p, filters)]
        return [SearchResult(id=p.id, score=1.0, payload=p.payload) for p in matches[:limit]]


def _matches(point: VectorPoint, filters: Sequence[EqualityFilter]) -> bool:
    return all(point.payload.get(f.field) == f.value for f in filters)


def _seed_knowledge(
    knowledge_storage: LocalFilesystemStorage, document_id: PaperId, chunk_count: int = 1
) -> list[str]:
    knowledge_storage.create_workspace(document_id)
    chunk_ids = [str(uuid4()) for _ in range(chunk_count)]
    knowledge_storage.write_json(
        document_id,
        "knowledge_units.json",
        {
            "document_id": str(document_id),
            "count": chunk_count,
            "chunks": [
                {
                    "id": chunk_id,
                    "paper_id": str(document_id),
                    "section_id": None,
                    "order": i,
                    "modality": ChunkModality.TEXT.value,
                    "text": f"chunk text {i}",
                    "asset_uri": None,
                    "token_count": None,
                    "source_element_ids": [],
                    "bounding_boxes": [],
                }
                for i, chunk_id in enumerate(chunk_ids)
            ],
        },
    )
    knowledge_storage.write_json(
        document_id,
        "relationships.json",
        {"document_id": str(document_id), "count": 0, "relationships": []},
    )
    return chunk_ids


def _seed_embeddings(
    embeddings_storage: LocalFilesystemStorage,
    document_id: PaperId,
    chunk_ids: list[str],
    model_version: str = "sha-1",
    target: str = "text",
) -> None:
    if not embeddings_storage.workspace_exists(document_id):
        embeddings_storage.create_workspace(document_id)
    embeddings_storage.write_json(
        document_id,
        "embeddings.json",
        {
            "document_id": str(document_id),
            "count": len(chunk_ids),
            "embeddings": [
                {
                    "embedding_id": str(uuid4()),
                    "knowledge_unit_id": chunk_id,
                    "paper_id": str(document_id),
                    "target": target,
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "model_name": "BAAI/bge-m3",
                    "model_version": model_version,
                    "embedding_dimension": 4,
                    "checksum": "abc",
                    "artifact_version": "1.0",
                    "source_representation_version": "repr-hash",
                    "created_at": datetime.now(UTC).isoformat(),
                }
                for chunk_id in chunk_ids
            ],
        },
    )
    embeddings_storage.write_json(
        document_id,
        "manifest.json",
        {
            "document_id": str(document_id),
            "model_name": "BAAI/bge-m3",
            "model_version": model_version,
            "embedding_dimension": 4,
            "artifact_version": "1.0",
            "source_representation_version": "repr-hash",
            "embedding_count": len(chunk_ids),
            "failed_count": 0,
            "skipped_image_count": 0,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )


@pytest.fixture
def storages(
    tmp_path: Path,
) -> tuple[LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage]:
    embeddings_storage = LocalFilesystemStorage(root=tmp_path / "embeddings")
    knowledge_storage = LocalFilesystemStorage(root=tmp_path / "knowledge")
    index_storage = LocalFilesystemStorage(root=tmp_path / "index")
    return embeddings_storage, knowledge_storage, index_storage


def _service(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage],
    vector_store: VectorStore,
) -> IndexingService:
    embeddings_storage, knowledge_storage, index_storage = storages
    repository = IndexRepository(
        embeddings_storage=embeddings_storage,
        knowledge_storage=knowledge_storage,
        index_storage=index_storage,
    )
    return IndexingService(
        repository=repository, vector_store=vector_store, collection_prefix="kimrag", batch_size=10
    )


def test_normal_indexing(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    embeddings_storage, knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    chunk_ids = _seed_knowledge(knowledge_storage, document_id, chunk_count=3)
    _seed_embeddings(embeddings_storage, document_id, chunk_ids)
    service = _service(storages, _FakeVectorStore())

    result = service.index_document(document_id)

    assert result.newly_indexed == 3
    assert result.manifest.indexed_vectors == 3
    assert result.manifest.failed_vectors == 0
    assert result.manifest.embedding_model == "BAAI/bge-m3"


def test_invalid_document_raises(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    service = _service(storages, _FakeVectorStore())

    with pytest.raises(EmbeddingArtifactsNotFoundError):
        service.index_document(PaperId(uuid4()))


def test_second_call_without_force_skips_reindexing(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    embeddings_storage, knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    chunk_ids = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_ids)
    vector_store = _FakeVectorStore()
    service = _service(storages, vector_store)

    service.index_document(document_id)
    calls_after_first = vector_store.upsert_call_count
    result = service.index_document(document_id)

    assert result.newly_indexed == 0
    assert result.manifest.indexed_vectors == 1  # existing count still reported
    assert vector_store.upsert_call_count == calls_after_first


def test_force_reindexes_even_when_fresh(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    embeddings_storage, knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    chunk_ids = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_ids)
    vector_store = _FakeVectorStore()
    service = _service(storages, vector_store)

    service.index_document(document_id)
    calls_after_first = vector_store.upsert_call_count
    result = service.index_document(document_id, force=True)

    assert result.newly_indexed == 1
    assert vector_store.upsert_call_count > calls_after_first


def test_embedding_change_triggers_reindexing(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    embeddings_storage, knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    chunk_ids = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_ids, model_version="sha-1")
    service = _service(storages, _FakeVectorStore())
    service.index_document(document_id)

    _seed_embeddings(embeddings_storage, document_id, chunk_ids, model_version="sha-2")
    result = service.index_document(document_id)

    assert result.newly_indexed == 1
    assert result.manifest.embedding_version == "sha-2"


def test_persistence_writes_index_manifest(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage],
    tmp_path: Path,
) -> None:
    embeddings_storage, knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    chunk_ids = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_ids)
    service = _service(storages, _FakeVectorStore())

    service.index_document(document_id)

    manifest_payload = json.loads(
        (tmp_path / "index" / str(document_id) / "index_manifest.json").read_text()
    )
    assert manifest_payload["document_id"] == str(document_id)
    assert manifest_payload["indexed_vectors"] == 1
    assert manifest_payload["distance_metric"] == "cosine"


def test_transient_upsert_failure_recovers_via_retry(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    embeddings_storage, knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    chunk_ids = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_ids)
    vector_store = _FakeVectorStore(fail_first_n_upserts=2)
    service = _service(storages, vector_store)

    result = service.index_document(document_id)

    assert result.newly_indexed == 1
    assert vector_store.upsert_call_count == 3


def test_all_upserts_failing_raises_no_vectors_indexed(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    embeddings_storage, knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    chunk_ids = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_ids)
    vector_store = _FakeVectorStore(fail_first_n_upserts=100)
    service = _service(storages, vector_store)

    with pytest.raises(NoVectorsIndexedError):
        service.index_document(document_id)


def test_multiple_targets_for_one_document_raise_not_supported(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    embeddings_storage, knowledge_storage, _ = storages
    document_id = PaperId(uuid4())
    chunk_ids = _seed_knowledge(knowledge_storage, document_id, chunk_count=2)
    embeddings_storage.create_workspace(document_id)
    embeddings_storage.write_json(
        document_id,
        "embeddings.json",
        {
            "document_id": str(document_id),
            "count": 2,
            "embeddings": [
                {
                    "embedding_id": str(uuid4()),
                    "knowledge_unit_id": chunk_ids[0],
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
                },
                {
                    "embedding_id": str(uuid4()),
                    "knowledge_unit_id": chunk_ids[1],
                    "paper_id": str(document_id),
                    "target": "image",
                    "vector": [0.1, 0.2],
                    "model_name": "clip-fake",
                    "model_version": "sha-9",
                    "embedding_dimension": 2,
                    "checksum": "def",
                    "artifact_version": "1.0",
                    "source_representation_version": "repr-hash",
                    "created_at": datetime.now(UTC).isoformat(),
                },
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
            "embedding_count": 2,
            "failed_count": 0,
            "skipped_image_count": 0,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    service = _service(storages, _FakeVectorStore())

    with pytest.raises(MultiCollectionIndexingNotSupportedError):
        service.index_document(document_id)


def test_provider_replacement_produces_identical_business_outcome(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    """Swapping the fake VectorStore for the real Qdrant provider changes
    nothing about the service's own behavior -- same counts, same manifest shape.

    Both runs use a unique model_version, so each gets its own
    freshly-created, disposable collection -- no cross-test-run
    accumulation in the shared Qdrant instance.
    """
    embeddings_storage, knowledge_storage, _ = storages
    shared_model_version = f"sha-{uuid4().hex}"

    document_id_fake = PaperId(uuid4())
    chunk_ids_fake = _seed_knowledge(knowledge_storage, document_id_fake, chunk_count=2)
    _seed_embeddings(
        embeddings_storage, document_id_fake, chunk_ids_fake, model_version=shared_model_version
    )
    fake_result = _service(storages, _FakeVectorStore()).index_document(document_id_fake)

    document_id_real = PaperId(uuid4())
    chunk_ids_real = _seed_knowledge(knowledge_storage, document_id_real, chunk_count=2)
    _seed_embeddings(
        embeddings_storage, document_id_real, chunk_ids_real, model_version=shared_model_version
    )
    real_provider = QdrantProvider(url="http://localhost:6333")
    real_result = _service(storages, real_provider).index_document(document_id_real)

    try:
        assert fake_result.newly_indexed == real_result.newly_indexed == 2
        assert fake_result.manifest.indexed_vectors == real_result.manifest.indexed_vectors
        assert fake_result.manifest.collection_name == real_result.manifest.collection_name
    finally:
        real_provider._client.delete_collection(
            real_result.manifest.collection_name
        )  # test-only cleanup
