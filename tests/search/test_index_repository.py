"""Tests for reading artifacts and staleness hashing, using real
LocalFilesystemStorage (against tmp_path). No Qdrant needed here."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from backend.domain import ChunkModality, PaperId
from backend.search.exceptions import EmbeddingArtifactsNotFoundError
from backend.search.models import DistanceMetric, IndexManifest
from backend.search.repository.index_repository import IndexRepository
from backend.storage.local_filesystem import LocalFilesystemStorage


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
    return chunk_id


def _seed_embeddings(
    embeddings_storage: LocalFilesystemStorage,
    document_id: PaperId,
    chunk_id: str,
    model_version: str = "sha-1",
) -> None:
    if not embeddings_storage.workspace_exists(document_id):
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
                    "model_version": model_version,
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
            "model_version": model_version,
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
def repository(
    tmp_path: Path,
) -> tuple[IndexRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage]:
    embeddings_storage = LocalFilesystemStorage(root=tmp_path / "embeddings")
    knowledge_storage = LocalFilesystemStorage(root=tmp_path / "knowledge")
    index_storage = LocalFilesystemStorage(root=tmp_path / "index")
    return (
        IndexRepository(
            embeddings_storage=embeddings_storage,
            knowledge_storage=knowledge_storage,
            index_storage=index_storage,
        ),
        embeddings_storage,
        knowledge_storage,
        index_storage,
    )


def _manifest(document_id: PaperId, source_hash: str) -> IndexManifest:
    return IndexManifest(
        document_id=document_id,
        collection_name="kimrag_baai_bge_m3_sha1_text",
        vector_dimension=4,
        distance_metric=DistanceMetric.COSINE,
        embedding_model="BAAI/bge-m3",
        embedding_version="sha-1",
        artifact_version="1.0",
        source_embedding_manifest=source_hash,
        checksum="checksum-1",
        indexed_vectors=1,
        failed_vectors=0,
        created_at=datetime.now(UTC),
    )


def test_read_embedding_artifacts_raises_for_missing_document(
    repository: tuple[
        IndexRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, _, _, _ = repository

    with pytest.raises(EmbeddingArtifactsNotFoundError):
        repo.read_embedding_artifacts(PaperId(uuid4()))


def test_read_embedding_artifacts_returns_seeded_data(
    repository: tuple[
        IndexRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, embeddings_storage, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    chunk_id = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_id)

    artifacts = repo.read_embedding_artifacts(document_id)

    assert len(artifacts) == 1
    assert str(artifacts[0].knowledge_unit_id) == chunk_id


def test_read_chunks_returns_chunks_keyed_by_id(
    repository: tuple[
        IndexRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, _, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    chunk_id = _seed_knowledge(knowledge_storage, document_id)

    chunks = repo.read_chunks(document_id)

    assert str(next(iter(chunks.keys()))) == chunk_id


def test_embedding_manifest_hash_is_stable_for_same_content(
    repository: tuple[
        IndexRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, embeddings_storage, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    chunk_id = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_id)

    first = repo.compute_embedding_manifest_hash(document_id)
    second = repo.compute_embedding_manifest_hash(document_id)

    assert first == second


def test_is_stale_when_no_index_manifest_exists(
    repository: tuple[
        IndexRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, embeddings_storage, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    chunk_id = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_id)

    assert repo.is_stale(document_id) is True


def test_is_not_stale_when_manifest_matches_current_hash(
    repository: tuple[
        IndexRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, embeddings_storage, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    chunk_id = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_id)
    current_hash = repo.compute_embedding_manifest_hash(document_id)
    repo.save_index_manifest(document_id, _manifest(document_id, current_hash))

    assert repo.is_stale(document_id) is False


def test_is_stale_when_embedding_manifest_changed(
    repository: tuple[
        IndexRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, embeddings_storage, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    chunk_id = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_id, model_version="sha-1")
    current_hash = repo.compute_embedding_manifest_hash(document_id)
    repo.save_index_manifest(document_id, _manifest(document_id, current_hash))

    _seed_embeddings(embeddings_storage, document_id, chunk_id, model_version="sha-2")

    assert repo.is_stale(document_id) is True


def test_save_and_load_index_manifest_round_trips(
    repository: tuple[
        IndexRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, embeddings_storage, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    chunk_id = _seed_knowledge(knowledge_storage, document_id)
    _seed_embeddings(embeddings_storage, document_id, chunk_id)
    manifest = _manifest(document_id, repo.compute_embedding_manifest_hash(document_id))

    repo.save_index_manifest(document_id, manifest)
    loaded = repo.load_index_manifest(document_id)

    assert loaded == manifest


def test_load_index_manifest_returns_none_when_absent(
    repository: tuple[
        IndexRepository, LocalFilesystemStorage, LocalFilesystemStorage, LocalFilesystemStorage
    ],
) -> None:
    repo, _, _, _ = repository

    assert repo.load_index_manifest(PaperId(uuid4())) is None
