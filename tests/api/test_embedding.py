"""End-to-end tests for the embedding API.

Overrides the text embedding provider with a fake -- this test verifies
routing, dependency wiring, and status-code mapping, not the real model
(covered separately in tests/embeddings/test_sentence_transformers_provider.py).
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import (
    get_embeddings_storage,
    get_knowledge_storage,
    get_text_embedding_provider,
)
from backend.domain import ChunkModality, PaperId
from backend.embeddings.interfaces.embedding_provider import EmbeddingProvider
from backend.storage.local_filesystem import LocalFilesystemStorage


class _FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def model_version(self) -> str:
        return "fake-v1"

    @property
    def embedding_dimension(self) -> int:
        return 4

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _seed_representation(
    knowledge_storage: LocalFilesystemStorage, document_id: PaperId, chunk_count: int
) -> None:
    knowledge_storage.create_workspace(document_id)
    chunks = [
        {
            "id": str(uuid4()),
            "paper_id": str(document_id),
            "section_id": None,
            "order": i,
            "modality": ChunkModality.TEXT.value,
            "text": f"chunk number {i}",
            "asset_uri": None,
            "token_count": None,
            "source_element_ids": [],
            "bounding_boxes": [],
        }
        for i in range(chunk_count)
    ]
    knowledge_storage.write_json(
        document_id,
        "knowledge_units.json",
        {"document_id": str(document_id), "count": chunk_count, "chunks": chunks},
    )


@pytest.fixture
def client_and_knowledge_storage(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, LocalFilesystemStorage]]:
    app = create_app()
    knowledge_storage = LocalFilesystemStorage(root=tmp_path / "knowledge")
    app.dependency_overrides[get_knowledge_storage] = lambda: knowledge_storage
    app.dependency_overrides[get_embeddings_storage] = lambda: LocalFilesystemStorage(
        root=tmp_path / "embeddings"
    )
    app.dependency_overrides[get_text_embedding_provider] = lambda: _FakeEmbeddingProvider()
    with TestClient(app) as test_client:
        yield test_client, knowledge_storage


def test_embed_document_returns_counts_and_model(
    client_and_knowledge_storage: tuple[TestClient, LocalFilesystemStorage],
) -> None:
    client, knowledge_storage = client_and_knowledge_storage
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, chunk_count=3)

    response = client.post(f"/documents/{document_id}/embed")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == str(document_id)
    assert body["embeddings_generated"] == 3
    assert body["model"] == "fake-model"
    assert body["status"] == "EMBEDDED"


def test_embed_document_returns_404_for_unrepresented_document(
    client_and_knowledge_storage: tuple[TestClient, LocalFilesystemStorage],
) -> None:
    client, _ = client_and_knowledge_storage

    response = client.post(f"/documents/{uuid4()}/embed")

    assert response.status_code == 404


def test_second_call_is_idempotent_without_force(
    client_and_knowledge_storage: tuple[TestClient, LocalFilesystemStorage],
) -> None:
    client, knowledge_storage = client_and_knowledge_storage
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, chunk_count=2)

    first = client.post(f"/documents/{document_id}/embed")
    second = client.post(f"/documents/{document_id}/embed")

    assert first.json()["embeddings_generated"] == 2
    assert second.json()["embeddings_generated"] == 2  # existing count, not recomputed


def test_force_query_param_regenerates(
    client_and_knowledge_storage: tuple[TestClient, LocalFilesystemStorage],
) -> None:
    client, knowledge_storage = client_and_knowledge_storage
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, chunk_count=1)

    client.post(f"/documents/{document_id}/embed")
    response = client.post(f"/documents/{document_id}/embed", params={"force": "true"})

    assert response.status_code == 200
    assert response.json()["embeddings_generated"] == 1
