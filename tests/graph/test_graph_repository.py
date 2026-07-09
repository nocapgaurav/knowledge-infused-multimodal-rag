"""Tests for reading the knowledge representation and staleness hashing,
using real LocalFilesystemStorage (against tmp_path). No Neo4j needed here."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from backend.domain import ChunkModality, PaperId
from backend.graph.exceptions import KnowledgeRepresentationNotFoundError
from backend.graph.models import GraphManifest
from backend.graph.repository.graph_repository import GraphRepository
from backend.storage.local_filesystem import LocalFilesystemStorage


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
def repository(
    tmp_path: Path,
) -> tuple[GraphRepository, LocalFilesystemStorage, LocalFilesystemStorage]:
    knowledge_storage = LocalFilesystemStorage(root=tmp_path / "knowledge")
    graph_storage = LocalFilesystemStorage(root=tmp_path / "graph")
    return (
        GraphRepository(knowledge_storage=knowledge_storage, graph_storage=graph_storage),
        knowledge_storage,
        graph_storage,
    )


def _manifest(document_id: PaperId, source_hash: str) -> GraphManifest:
    return GraphManifest(
        document_id=document_id,
        artifact_version="1.0",
        graph_version="1.0",
        node_count=2,
        relationship_count=1,
        checksum="checksum-1",
        source_representation_version=source_hash,
        created_at=datetime.now(UTC),
    )


def test_read_chunks_raises_for_missing_document(
    repository: tuple[GraphRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, _, _ = repository

    with pytest.raises(KnowledgeRepresentationNotFoundError):
        repo.read_chunks(PaperId(uuid4()))


def test_read_chunks_returns_seeded_chunks(
    repository: tuple[GraphRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    chunk_id = _seed_knowledge(knowledge_storage, document_id)

    chunks = repo.read_chunks(document_id)

    assert len(chunks) == 1
    assert str(chunks[0].id) == chunk_id


def test_read_relationships_returns_empty_list_when_none_recorded(
    repository: tuple[GraphRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id)

    assert repo.read_relationships(document_id) == []


def test_representation_hash_is_stable_for_identical_content(
    repository: tuple[GraphRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id)

    first = repo.compute_representation_hash(document_id)
    second = repo.compute_representation_hash(document_id)

    assert first == second


def test_representation_hash_changes_when_content_changes(
    repository: tuple[GraphRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id, text="version one")
    first = repo.compute_representation_hash(document_id)

    _seed_knowledge(knowledge_storage, document_id, text="version two")
    second = repo.compute_representation_hash(document_id)

    assert first != second


def test_save_and_load_graph_manifest_round_trips(
    repository: tuple[GraphRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, knowledge_storage, _ = repository
    document_id = PaperId(uuid4())
    _seed_knowledge(knowledge_storage, document_id)
    manifest = _manifest(document_id, repo.compute_representation_hash(document_id))

    repo.save_graph_manifest(document_id, manifest)
    loaded = repo.load_graph_manifest(document_id)

    assert loaded == manifest


def test_load_graph_manifest_returns_none_when_absent(
    repository: tuple[GraphRepository, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    repo, _, _ = repository

    assert repo.load_graph_manifest(PaperId(uuid4())) is None
