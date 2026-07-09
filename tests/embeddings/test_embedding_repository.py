"""Tests for embedding persistence and staleness hashing, using real
LocalFilesystemStorage (against tmp_path)."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from backend.domain import ChunkId, PaperId
from backend.embeddings.exceptions import RepresentationNotFoundError
from backend.embeddings.models import (
    EmbeddingArtifact,
    EmbeddingId,
    EmbeddingManifest,
    EmbeddingTarget,
)
from backend.embeddings.repository.embedding_repository import EmbeddingRepository
from backend.storage.local_filesystem import LocalFilesystemStorage


def _seed_representation(
    knowledge_storage: LocalFilesystemStorage, document_id: PaperId, text: str
) -> None:
    if not knowledge_storage.workspace_exists(document_id):
        knowledge_storage.create_workspace(document_id)
    knowledge_storage.write_json(
        document_id,
        "knowledge_units.json",
        {
            "document_id": str(document_id),
            "count": 1,
            "chunks": [
                {
                    "id": str(uuid4()),
                    "paper_id": str(document_id),
                    "section_id": None,
                    "order": 0,
                    "modality": "text",
                    "text": text,
                    "asset_uri": None,
                    "token_count": None,
                    "source_element_ids": [],
                    "bounding_boxes": [],
                }
            ],
        },
    )


@pytest.fixture
def repository(
    tmp_path: Path,
) -> tuple[EmbeddingRepository, LocalFilesystemStorage, LocalFilesystemStorage]:
    knowledge_storage = LocalFilesystemStorage(root=tmp_path / "knowledge")
    embeddings_storage = LocalFilesystemStorage(root=tmp_path / "embeddings")
    return (
        EmbeddingRepository(
            knowledge_storage=knowledge_storage, embeddings_storage=embeddings_storage
        ),
        knowledge_storage,
        embeddings_storage,
    )


def _manifest(
    document_id: PaperId, representation_version: str, model_version: str = "sha-1"
) -> EmbeddingManifest:
    return EmbeddingManifest(
        document_id=document_id,
        model_name="BAAI/bge-m3",
        model_version=model_version,
        embedding_dimension=1024,
        artifact_version="1.0",
        source_representation_version=representation_version,
        embedding_count=1,
        failed_count=0,
        skipped_image_count=0,
        created_at=datetime.now(UTC),
    )


def test_read_knowledge_units_payload_raises_for_unrepresented_document(
    repository: tuple[EmbeddingRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, _, _ = repository

    with pytest.raises(RepresentationNotFoundError):
        repo.read_knowledge_units_payload(PaperId(uuid4()))


def test_representation_version_is_stable_for_same_content(
    repository: tuple[EmbeddingRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, "stable text")

    first = repo.compute_representation_version(document_id)
    second = repo.compute_representation_version(document_id)

    assert first == second


def test_representation_version_changes_when_content_changes(
    repository: tuple[EmbeddingRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, knowledge_storage, _ = repository
    document_a = PaperId(uuid4())
    document_b = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_a, "text A")
    _seed_representation(knowledge_storage, document_b, "text B")

    assert repo.compute_representation_version(document_a) != repo.compute_representation_version(
        document_b
    )


def test_load_manifest_returns_none_when_none_exists(
    repository: tuple[EmbeddingRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, _, _ = repository

    assert repo.load_manifest(PaperId(uuid4())) is None


def test_save_and_load_manifest_round_trips(
    repository: tuple[EmbeddingRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, "text")
    representation_version = repo.compute_representation_version(document_id)
    manifest = _manifest(document_id, representation_version)
    artifact = EmbeddingArtifact(
        embedding_id=EmbeddingId(uuid4()),
        knowledge_unit_id=ChunkId(uuid4()),
        paper_id=document_id,
        target=EmbeddingTarget.TEXT,
        vector=[0.1, 0.2],
        model_name="BAAI/bge-m3",
        model_version="sha-1",
        embedding_dimension=2,
        checksum="abc",
        artifact_version="1.0",
        source_representation_version=representation_version,
        created_at=datetime.now(UTC),
    )

    repo.save(document_id, [artifact], manifest)
    loaded = repo.load_manifest(document_id)

    assert loaded == manifest


def test_is_stale_when_no_manifest_exists(
    repository: tuple[EmbeddingRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, "text")

    assert repo.is_stale(document_id, "BAAI/bge-m3", "sha-1") is True


def test_is_stale_when_model_changed(
    repository: tuple[EmbeddingRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, "text")
    representation_version = repo.compute_representation_version(document_id)
    repo.save(
        document_id, [], _manifest(document_id, representation_version, model_version="old-sha")
    )

    assert repo.is_stale(document_id, "BAAI/bge-m3", "new-sha") is True


def test_is_stale_when_representation_changed(
    repository: tuple[EmbeddingRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, "original text")
    representation_version = repo.compute_representation_version(document_id)
    repo.save(document_id, [], _manifest(document_id, representation_version))

    _seed_representation(knowledge_storage, document_id, "changed text")

    assert repo.is_stale(document_id, "BAAI/bge-m3", "sha-1") is True


def test_is_not_stale_when_nothing_changed(
    repository: tuple[EmbeddingRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, "text")
    representation_version = repo.compute_representation_version(document_id)
    repo.save(document_id, [], _manifest(document_id, representation_version))

    assert repo.is_stale(document_id, "BAAI/bge-m3", "sha-1") is False
