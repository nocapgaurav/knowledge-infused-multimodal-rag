"""Exceptions raised by the knowledge representation engine."""

from backend.domain import PaperId, RelationshipId


class ChunkingError(Exception):
    """Base class for all knowledge representation errors."""


class PaperNotParsedError(ChunkingError):
    """Raised when no parsed `Paper` artifact exists for a given document id."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"no parsed document found for {document_id}")


class RepresentationValidationError(ChunkingError):
    """Base class for structural defects found in a freshly built representation."""


class EmptyRepresentationError(RepresentationValidationError):
    """Raised when a paper produces no knowledge units at all."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"document {document_id} produced no knowledge units")


class DuplicateChunkOrderError(RepresentationValidationError):
    """Raised when two knowledge units share the same `order` value."""

    def __init__(self, *, document_id: PaperId, order: int) -> None:
        self.document_id = document_id
        self.order = order
        super().__init__(f"document {document_id} has two knowledge units with order {order}")


class InvalidChunkReferenceError(RepresentationValidationError):
    """Raised when a knowledge unit references a paper or section it does not belong to."""

    def __init__(self, *, document_id: PaperId, reason: str) -> None:
        self.document_id = document_id
        self.reason = reason
        super().__init__(
            f"document {document_id} has an invalid knowledge unit reference: {reason}"
        )


class DanglingRelationshipError(RepresentationValidationError):
    """Raised when a relationship points at a chunk that does not exist."""

    def __init__(self, *, document_id: PaperId, relationship_id: RelationshipId) -> None:
        self.document_id = document_id
        self.relationship_id = relationship_id
        super().__init__(
            f"document {document_id} has relationship {relationship_id} "
            "referencing a nonexistent chunk"
        )


class RepresentationStorageError(ChunkingError):
    """Raised when a storage failure prevents knowledge representation artifacts
    from being persisted."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(
            f"a storage error occurred while persisting the knowledge representation "
            f"for document {document_id}"
        )
