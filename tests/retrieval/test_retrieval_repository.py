"""Tests for reading upstream manifests and persisting retrieval manifests,
using real LocalFilesystemStorage (against tmp_path). No Qdrant/Neo4j needed here."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from backend.domain import PaperId
from backend.retrieval.exceptions import DocumentNotGraphedError, DocumentNotIndexedError
from backend.retrieval.models import RetrievalManifest, RetrievalStatistics
from backend.retrieval.repository.retrieval_repository import RetrievalRepository
from backend.storage.local_filesystem import LocalFilesystemStorage


@pytest.fixture
def repository(
    tmp_path: Path,
) -> tuple[
    RetrievalRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
]:
    embeddings_storage = LocalFilesystemStorage(root=tmp_path / "embeddings")
    index_storage = LocalFilesystemStorage(root=tmp_path / "index")
    graph_storage = LocalFilesystemStorage(root=tmp_path / "graph")
    retrieval_storage = LocalFilesystemStorage(root=tmp_path / "retrieval")
    return (
        RetrievalRepository(
            embeddings_storage=embeddings_storage,
            index_storage=index_storage,
            graph_storage=graph_storage,
            retrieval_storage=retrieval_storage,
        ),
        embeddings_storage,
        index_storage,
        graph_storage,
    )


def _seed_embedding_manifest(storage: LocalFilesystemStorage, document_id: PaperId) -> None:
    storage.create_workspace(document_id)
    storage.write_json(
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


def _seed_index_manifest(
    storage: LocalFilesystemStorage, document_id: PaperId, collection_name: str
) -> None:
    storage.create_workspace(document_id)
    storage.write_json(
        document_id,
        "index_manifest.json",
        {
            "document_id": str(document_id),
            "collection_name": collection_name,
            "vector_dimension": 4,
            "distance_metric": "cosine",
            "embedding_model": "BAAI/bge-m3",
            "embedding_version": "sha-1",
            "artifact_version": "1.0",
            "source_embedding_manifest": "manifest-hash",
            "checksum": "checksum-1",
            "indexed_vectors": 1,
            "failed_vectors": 0,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )


def _seed_graph_manifest(storage: LocalFilesystemStorage, document_id: PaperId) -> None:
    storage.create_workspace(document_id)
    storage.write_json(
        document_id,
        "graph_manifest.json",
        {
            "document_id": str(document_id),
            "artifact_version": "1.0",
            "graph_version": "1.0",
            "node_count": 2,
            "relationship_count": 1,
            "checksum": "graph-checksum",
            "source_representation_version": "repr-hash",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )


def test_resolve_collection_returns_index_manifest_collection_name(
    repository: tuple[
        RetrievalRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, _, index_storage, _ = repository
    document_id = PaperId(uuid4())
    _seed_index_manifest(index_storage, document_id, "kimrag_bge_m3_text")

    assert repo.resolve_collection(document_id) == "kimrag_bge_m3_text"


def test_resolve_collection_raises_when_not_indexed(
    repository: tuple[
        RetrievalRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, _, _, _ = repository

    with pytest.raises(DocumentNotIndexedError):
        repo.resolve_collection(PaperId(uuid4()))


def test_read_embedding_manifest_returns_seeded_data(
    repository: tuple[
        RetrievalRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, embeddings_storage, _, _ = repository
    document_id = PaperId(uuid4())
    _seed_embedding_manifest(embeddings_storage, document_id)

    manifest = repo.read_embedding_manifest(document_id)

    assert manifest.model_version == "sha-1"
    assert manifest.source_representation_version == "repr-hash"


def test_read_graph_manifest_raises_when_not_graphed(
    repository: tuple[
        RetrievalRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, _, _, _ = repository

    with pytest.raises(DocumentNotGraphedError):
        repo.read_graph_manifest(PaperId(uuid4()))


def test_read_graph_manifest_returns_seeded_data(
    repository: tuple[
        RetrievalRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, _, _, graph_storage = repository
    document_id = PaperId(uuid4())
    _seed_graph_manifest(graph_storage, document_id)

    manifest = repo.read_graph_manifest(document_id)

    assert manifest.graph_version == "1.0"


def test_save_and_persist_retrieval_manifest(
    repository: tuple[
        RetrievalRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
    tmp_path: Path,
) -> None:
    repo, _, _, _ = repository
    document_id = PaperId(uuid4())
    manifest = RetrievalManifest(
        document_id=document_id,
        query="what is the result?",
        retrieval_version="1.0",
        retrieval_strategy_version="1.0",
        representation_version="repr-hash",
        embedding_version="sha-1",
        graph_version="1.0",
        statistics=RetrievalStatistics(
            candidates_generated=5,
            candidates_expanded=3,
            candidates_scored=8,
            evidence_groups=2,
            evidence_items=4,
            duration_ms=42.0,
        ),
        created_at=datetime.now(UTC),
    )

    repo.save_retrieval_manifest(document_id, manifest)

    saved_path = tmp_path / "retrieval" / str(document_id) / "retrieval_manifest.json"
    assert saved_path.exists()


def test_save_retrieval_manifest_overwrites_previous_run(
    repository: tuple[
        RetrievalRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, _, _, _ = repository
    document_id = PaperId(uuid4())

    def _manifest(query: str) -> RetrievalManifest:
        return RetrievalManifest(
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
                duration_ms=1.0,
            ),
            created_at=datetime.now(UTC),
        )

    repo.save_retrieval_manifest(document_id, _manifest("first query"))
    repo.save_retrieval_manifest(document_id, _manifest("second query"))

    # no error re-creating the workspace, and the file reflects the latest run
    assert repo._retrieval_storage.read_json(document_id, "retrieval_manifest.json")["query"] == (
        "second query"
    )
