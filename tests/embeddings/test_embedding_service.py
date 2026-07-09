"""Tests for embedding service orchestration: staleness, retry, partial
success, persistence -- using a fake EmbeddingProvider and real
LocalFilesystemStorage (against tmp_path)."""

import json
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pytest

from backend.domain import ChunkModality, PaperId
from backend.embeddings.exceptions import NoEmbeddingsProducedError, RepresentationNotFoundError
from backend.embeddings.interfaces.embedding_provider import EmbeddingProvider
from backend.embeddings.repository.embedding_repository import EmbeddingRepository
from backend.embeddings.services.embedding_service import EmbeddingService
from backend.storage.local_filesystem import LocalFilesystemStorage


class _FakeEmbeddingProvider(EmbeddingProvider):
    """A deterministic, fast stand-in for a real embedding model."""

    def __init__(
        self,
        dimension: int = 4,
        model_name: str = "fake-model",
        model_version: str = "v1",
        fail_on: frozenset[str] = frozenset(),
        fail_first_n_calls: int = 0,
    ) -> None:
        self._dimension = dimension
        self._model_name = model_name
        self._model_version = model_version
        self._fail_on = fail_on
        self._fail_first_n_calls = fail_first_n_calls
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        from backend.embeddings.exceptions import EmbeddingProviderError

        self.call_count += 1
        if self.call_count <= self._fail_first_n_calls:
            raise EmbeddingProviderError(reason="simulated transient failure")
        if any(text in self._fail_on for text in texts):
            raise EmbeddingProviderError(reason="simulated permanent failure")
        return [[float(len(text) % 7)] * self._dimension for text in texts]


def _chunk_payload(text: str, asset_uri: str | None = None) -> dict:
    return {
        "id": str(uuid4()),
        "paper_id": None,  # filled in by _seed
        "section_id": None,
        "order": 0,
        "modality": ChunkModality.TEXT.value,
        "text": text,
        "asset_uri": asset_uri,
        "token_count": None,
        "source_element_ids": [],
        "bounding_boxes": [],
    }


def _seed_representation(
    knowledge_storage: LocalFilesystemStorage, document_id: PaperId, texts: list[str]
) -> None:
    if not knowledge_storage.workspace_exists(document_id):
        knowledge_storage.create_workspace(document_id)
    chunks = []
    for i, text in enumerate(texts):
        payload = _chunk_payload(text)
        payload["paper_id"] = str(document_id)
        payload["order"] = i
        chunks.append(payload)
    knowledge_storage.write_json(
        document_id,
        "knowledge_units.json",
        {"document_id": str(document_id), "count": len(chunks), "chunks": chunks},
    )


@pytest.fixture
def storages(tmp_path: Path) -> tuple[LocalFilesystemStorage, LocalFilesystemStorage]:
    knowledge_storage = LocalFilesystemStorage(root=tmp_path / "knowledge")
    embeddings_storage = LocalFilesystemStorage(root=tmp_path / "embeddings")
    return knowledge_storage, embeddings_storage


def _service(
    knowledge_storage: LocalFilesystemStorage,
    embeddings_storage: LocalFilesystemStorage,
    provider: EmbeddingProvider,
) -> EmbeddingService:
    repository = EmbeddingRepository(
        knowledge_storage=knowledge_storage, embeddings_storage=embeddings_storage
    )
    return EmbeddingService(repository=repository, text_provider=provider, batch_size=10)


def test_normal_embedding_generation(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage], tmp_path: Path
) -> None:
    knowledge_storage, embeddings_storage = storages
    document_id = PaperId(uuid4())
    _seed_representation(
        knowledge_storage, document_id, ["first chunk", "second chunk", "third chunk"]
    )
    service = _service(knowledge_storage, embeddings_storage, _FakeEmbeddingProvider())

    result = service.embed_document(document_id)

    assert len(result.artifacts) == 3
    assert result.manifest.embedding_count == 3
    assert result.manifest.failed_count == 0
    assert result.manifest.model_name == "fake-model"


def test_empty_knowledge_representation_raises(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    knowledge_storage, embeddings_storage = storages
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, [])
    service = _service(knowledge_storage, embeddings_storage, _FakeEmbeddingProvider())

    with pytest.raises(NoEmbeddingsProducedError):
        service.embed_document(document_id)


def test_invalid_document_raises_representation_not_found(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    knowledge_storage, embeddings_storage = storages
    service = _service(knowledge_storage, embeddings_storage, _FakeEmbeddingProvider())

    with pytest.raises(RepresentationNotFoundError):
        service.embed_document(PaperId(uuid4()))


def test_second_call_without_force_skips_regeneration(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    knowledge_storage, embeddings_storage = storages
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, ["a chunk"])
    provider = _FakeEmbeddingProvider()
    service = _service(knowledge_storage, embeddings_storage, provider)

    first = service.embed_document(document_id)
    calls_after_first = provider.call_count
    second = service.embed_document(document_id)

    assert len(first.artifacts) == 1
    assert second.artifacts == []  # nothing regenerated
    assert second.manifest.embedding_count == 1  # existing count still reported
    assert provider.call_count == calls_after_first  # provider was not called again


def test_force_regenerates_even_when_fresh(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    knowledge_storage, embeddings_storage = storages
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, ["a chunk"])
    provider = _FakeEmbeddingProvider()
    service = _service(knowledge_storage, embeddings_storage, provider)

    service.embed_document(document_id)
    calls_after_first = provider.call_count
    result = service.embed_document(document_id, force=True)

    assert len(result.artifacts) == 1
    assert provider.call_count > calls_after_first


def test_representation_change_triggers_regeneration(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    knowledge_storage, embeddings_storage = storages
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, ["original"])
    service = _service(knowledge_storage, embeddings_storage, _FakeEmbeddingProvider())
    service.embed_document(document_id)

    _seed_representation(knowledge_storage, document_id, ["original", "and a new chunk"])
    result = service.embed_document(document_id)

    assert len(result.artifacts) == 2


def test_provider_replacement_does_not_change_business_logic(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    """Swapping the concrete provider changes only the recorded model
    metadata -- the service's own behavior is identical either way."""
    knowledge_storage, embeddings_storage = storages
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, ["a chunk", "another chunk"])

    result_a = _service(
        knowledge_storage,
        embeddings_storage,
        _FakeEmbeddingProvider(model_name="model-a", dimension=4),
    ).embed_document(document_id)

    document_id_b = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id_b, ["a chunk", "another chunk"])
    result_b = _service(
        knowledge_storage,
        embeddings_storage,
        _FakeEmbeddingProvider(model_name="model-b", dimension=8),
    ).embed_document(document_id_b)

    assert result_a.manifest.embedding_count == result_b.manifest.embedding_count == 2
    assert result_a.manifest.model_name == "model-a"
    assert result_b.manifest.model_name == "model-b"
    assert result_a.manifest.embedding_dimension == 4
    assert result_b.manifest.embedding_dimension == 8


def test_persistence_writes_expected_files(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage], tmp_path: Path
) -> None:
    knowledge_storage, embeddings_storage = storages
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, ["a chunk"])
    service = _service(knowledge_storage, embeddings_storage, _FakeEmbeddingProvider())

    service.embed_document(document_id)

    workspace = tmp_path / "embeddings" / str(document_id)
    embeddings_payload = json.loads((workspace / "embeddings.json").read_text())
    manifest_payload = json.loads((workspace / "manifest.json").read_text())

    assert embeddings_payload["count"] == 1
    assert len(embeddings_payload["embeddings"]) == 1
    assert manifest_payload["document_id"] == str(document_id)


def test_manifest_contains_expected_metadata(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    knowledge_storage, embeddings_storage = storages
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, ["a chunk"])
    service = _service(knowledge_storage, embeddings_storage, _FakeEmbeddingProvider())

    result = service.embed_document(document_id)

    assert result.manifest.artifact_version == "1.0"
    assert result.manifest.embedding_dimension == 4
    assert result.manifest.source_representation_version
    assert result.manifest.created_at is not None


def test_partial_failure_still_persists_successful_chunks(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    """With one chunk per batch, a failing chunk doesn't take down the
    others in the same document -- genuine partial success."""
    knowledge_storage, embeddings_storage = storages
    document_id = PaperId(uuid4())
    _seed_representation(
        knowledge_storage, document_id, ["good chunk one", "bad chunk", "good chunk two"]
    )
    provider = _FakeEmbeddingProvider(fail_on=frozenset({"bad chunk"}))
    repository = EmbeddingRepository(
        knowledge_storage=knowledge_storage, embeddings_storage=embeddings_storage
    )
    service = EmbeddingService(repository=repository, text_provider=provider, batch_size=1)

    result = service.embed_document(document_id)

    assert len(result.artifacts) == 2
    assert result.manifest.embedding_count == 2
    assert result.manifest.failed_count == 1


def test_all_chunks_failing_raises_no_embeddings_produced(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    knowledge_storage, embeddings_storage = storages
    document_id_bad = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id_bad, ["bad chunk"])
    provider = _FakeEmbeddingProvider(fail_on=frozenset({"bad chunk"}))
    service = _service(knowledge_storage, embeddings_storage, provider)

    with pytest.raises(NoEmbeddingsProducedError):
        service.embed_document(document_id_bad)


def test_transient_failure_recovers_via_retry(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    knowledge_storage, embeddings_storage = storages
    document_id = PaperId(uuid4())
    _seed_representation(knowledge_storage, document_id, ["a chunk"])
    provider = _FakeEmbeddingProvider(fail_first_n_calls=2)
    service = _service(knowledge_storage, embeddings_storage, provider)

    result = service.embed_document(document_id)

    assert len(result.artifacts) == 1
    assert provider.call_count == 3  # two failures, then success
