"""Tests for structural validation of a built knowledge representation."""

from uuid import uuid4

import pytest

from backend.chunking.exceptions import (
    DanglingRelationshipError,
    DuplicateChunkOrderError,
    EmptyRepresentationError,
    InvalidChunkReferenceError,
)
from backend.chunking.validator.knowledge_validator import validate_representation
from backend.domain import (
    Chunk,
    ChunkId,
    ChunkModality,
    Metadata,
    Paper,
    PaperId,
    Relationship,
    RelationshipType,
    SectionId,
)


def _metadata() -> Metadata:
    return Metadata(title="A Paper", source_filename="p.pdf")


def _chunk(paper_id: PaperId, order: int, section_id: SectionId | None = None) -> Chunk:
    return Chunk(
        paper_id=paper_id,
        section_id=section_id,
        order=order,
        modality=ChunkModality.TEXT,
        text="some text",
    )


def test_valid_representation_passes() -> None:
    paper_id = PaperId(uuid4())
    paper = Paper(id=paper_id, metadata=_metadata())
    chunks = [_chunk(paper_id, 0), _chunk(paper_id, 1)]

    validate_representation(paper, chunks, [])  # should not raise


def test_no_chunks_raises_empty_representation_error() -> None:
    paper_id = PaperId(uuid4())
    paper = Paper(id=paper_id, metadata=_metadata())

    with pytest.raises(EmptyRepresentationError):
        validate_representation(paper, [], [])


def test_duplicate_order_raises() -> None:
    paper_id = PaperId(uuid4())
    paper = Paper(id=paper_id, metadata=_metadata())
    chunks = [_chunk(paper_id, 0), _chunk(paper_id, 0)]

    with pytest.raises(DuplicateChunkOrderError):
        validate_representation(paper, chunks, [])


def test_chunk_from_a_different_paper_raises() -> None:
    paper_id = PaperId(uuid4())
    paper = Paper(id=paper_id, metadata=_metadata())
    chunk = _chunk(PaperId(uuid4()), 0)

    with pytest.raises(InvalidChunkReferenceError):
        validate_representation(paper, [chunk], [])


def test_chunk_referencing_unknown_section_raises() -> None:
    paper_id = PaperId(uuid4())
    paper = Paper(id=paper_id, metadata=_metadata())
    chunk = _chunk(paper_id, 0, section_id=SectionId(uuid4()))

    with pytest.raises(InvalidChunkReferenceError):
        validate_representation(paper, [chunk], [])


def test_dangling_relationship_raises() -> None:
    paper_id = PaperId(uuid4())
    paper = Paper(id=paper_id, metadata=_metadata())
    chunk = _chunk(paper_id, 0)
    dangling = Relationship(
        paper_id=paper_id,
        source_chunk_id=chunk.id,
        target_chunk_id=ChunkId(uuid4()),
        relationship_type=RelationshipType.CITES,
    )

    with pytest.raises(DanglingRelationshipError):
        validate_representation(paper, [chunk], [dangling])
