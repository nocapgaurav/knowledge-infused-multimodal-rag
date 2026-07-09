"""Tests for structural validation of a generated embedding set."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from backend.domain import Chunk as DomainChunk
from backend.domain import ChunkId, ChunkModality, PaperId
from backend.embeddings.exceptions import (
    DimensionMismatchError,
    DuplicateEmbeddingError,
    NoEmbeddingsProducedError,
    UnknownKnowledgeUnitError,
)
from backend.embeddings.models import EmbeddingArtifact, EmbeddingId, EmbeddingTarget
from backend.embeddings.validator.embedding_validator import validate_embeddings


def _chunk(paper_id: PaperId, order: int = 0) -> DomainChunk:
    return DomainChunk(paper_id=paper_id, order=order, modality=ChunkModality.TEXT, text="text")


def _artifact(
    paper_id: PaperId,
    knowledge_unit_id: ChunkId,
    target: EmbeddingTarget = EmbeddingTarget.TEXT,
    dimension: int = 3,
) -> EmbeddingArtifact:
    return EmbeddingArtifact(
        embedding_id=EmbeddingId(uuid4()),
        knowledge_unit_id=knowledge_unit_id,
        paper_id=paper_id,
        target=target,
        vector=[0.1] * dimension,
        model_name="BAAI/bge-m3",
        model_version="sha-1",
        embedding_dimension=dimension,
        checksum="abc",
        artifact_version="1.0",
        source_representation_version="hash-1",
        created_at=datetime.now(UTC),
    )


def test_valid_set_passes() -> None:
    paper_id = PaperId(uuid4())
    chunk = _chunk(paper_id)

    validate_embeddings(paper_id, [chunk], [_artifact(paper_id, chunk.id)])  # should not raise


def test_no_artifacts_raises() -> None:
    paper_id = PaperId(uuid4())
    chunk = _chunk(paper_id)

    with pytest.raises(NoEmbeddingsProducedError):
        validate_embeddings(paper_id, [chunk], [])


def test_artifact_for_unknown_chunk_raises() -> None:
    paper_id = PaperId(uuid4())
    chunk = _chunk(paper_id)
    dangling = _artifact(paper_id, ChunkId(uuid4()))

    with pytest.raises(UnknownKnowledgeUnitError):
        validate_embeddings(paper_id, [chunk], [dangling])


def test_duplicate_target_for_same_chunk_raises() -> None:
    paper_id = PaperId(uuid4())
    chunk = _chunk(paper_id)
    artifacts = [_artifact(paper_id, chunk.id), _artifact(paper_id, chunk.id)]

    with pytest.raises(DuplicateEmbeddingError):
        validate_embeddings(paper_id, [chunk], artifacts)


def test_same_chunk_can_have_both_text_and_image_targets() -> None:
    paper_id = PaperId(uuid4())
    chunk = _chunk(paper_id)
    artifacts = [
        _artifact(paper_id, chunk.id, target=EmbeddingTarget.TEXT),
        _artifact(paper_id, chunk.id, target=EmbeddingTarget.IMAGE),
    ]

    validate_embeddings(paper_id, [chunk], artifacts)  # should not raise


def test_inconsistent_dimension_for_same_model_and_target_raises() -> None:
    paper_id = PaperId(uuid4())
    chunk_a = _chunk(paper_id, order=0)
    chunk_b = _chunk(paper_id, order=1)
    artifacts = [
        _artifact(paper_id, chunk_a.id, dimension=3),
        _artifact(paper_id, chunk_b.id, dimension=4),
    ]

    with pytest.raises(DimensionMismatchError):
        validate_embeddings(paper_id, [chunk_a, chunk_b], artifacts)
