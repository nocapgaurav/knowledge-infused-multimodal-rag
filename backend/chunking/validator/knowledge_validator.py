"""Structural validation of a freshly built knowledge representation."""

from backend.chunking.exceptions import (
    DanglingRelationshipError,
    DuplicateChunkOrderError,
    EmptyRepresentationError,
    InvalidChunkReferenceError,
)
from backend.domain import Chunk, Paper, Relationship


def validate_representation(
    paper: Paper, chunks: list[Chunk], relationships: list[Relationship]
) -> None:
    """Validate a freshly built knowledge representation before it is persisted.

    Args:
        paper: The paper the representation was built from.
        chunks: Knowledge units produced by the builder.
        relationships: Relationships produced by the builder.

    Raises:
        EmptyRepresentationError: No chunks were produced at all.
        DuplicateChunkOrderError: Two chunks share the same `order` value.
        InvalidChunkReferenceError: A chunk's `paper_id` or `section_id` is inconsistent.
        DanglingRelationshipError: A relationship points at a chunk that does not exist.
    """
    _validate_not_empty(paper, chunks)
    _validate_unique_order(paper, chunks)
    _validate_chunk_ownership(paper, chunks)
    _validate_relationship_endpoints(paper, chunks, relationships)


def _validate_not_empty(paper: Paper, chunks: list[Chunk]) -> None:
    if not chunks:
        raise EmptyRepresentationError(document_id=paper.id)


def _validate_unique_order(paper: Paper, chunks: list[Chunk]) -> None:
    seen: set[int] = set()
    for chunk in chunks:
        if chunk.order in seen:
            raise DuplicateChunkOrderError(document_id=paper.id, order=chunk.order)
        seen.add(chunk.order)


def _validate_chunk_ownership(paper: Paper, chunks: list[Chunk]) -> None:
    section_ids = {section.id for section in paper.sections}
    for chunk in chunks:
        if chunk.paper_id != paper.id:
            raise InvalidChunkReferenceError(
                document_id=paper.id,
                reason=f"chunk {chunk.id} has paper_id {chunk.paper_id}, expected {paper.id}",
            )
        if chunk.section_id is not None and chunk.section_id not in section_ids:
            raise InvalidChunkReferenceError(
                document_id=paper.id,
                reason=f"chunk {chunk.id} references unknown section {chunk.section_id}",
            )


def _validate_relationship_endpoints(
    paper: Paper, chunks: list[Chunk], relationships: list[Relationship]
) -> None:
    chunk_ids = {chunk.id for chunk in chunks}
    for relationship in relationships:
        if (
            relationship.source_chunk_id not in chunk_ids
            or relationship.target_chunk_id not in chunk_ids
        ):
            raise DanglingRelationshipError(document_id=paper.id, relationship_id=relationship.id)
