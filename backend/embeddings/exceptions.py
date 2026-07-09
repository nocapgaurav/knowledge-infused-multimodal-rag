"""Exceptions raised by the embedding infrastructure."""

from backend.domain import PaperId


class EmbeddingError(Exception):
    """Base class for all embedding infrastructure errors."""


class RepresentationNotFoundError(EmbeddingError):
    """Raised when no knowledge representation exists for a given document id."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"no knowledge representation found for {document_id}")


class EmbeddingProviderError(EmbeddingError):
    """Raised by a provider when it fails to produce an embedding."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"embedding provider failed: {reason}")


class NoEmbeddingsProducedError(EmbeddingError):
    """Raised when every knowledge unit failed to embed."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"document {document_id} produced no successful embeddings")


class EmbeddingValidationError(EmbeddingError):
    """Base class for structural defects found in a freshly generated embedding set."""


class DimensionMismatchError(EmbeddingValidationError):
    """Raised when artifacts for the same model/target report different dimensions."""

    def __init__(self, *, document_id: PaperId, reason: str) -> None:
        self.document_id = document_id
        self.reason = reason
        super().__init__(f"document {document_id} has inconsistent embedding dimensions: {reason}")


class DuplicateEmbeddingError(EmbeddingValidationError):
    """Raised when the same (knowledge_unit_id, target) pair appears more than once."""

    def __init__(self, *, document_id: PaperId, reason: str) -> None:
        self.document_id = document_id
        self.reason = reason
        super().__init__(f"document {document_id} has duplicate embeddings: {reason}")


class UnknownKnowledgeUnitError(EmbeddingValidationError):
    """Raised when an artifact references a knowledge unit that does not exist."""

    def __init__(self, *, document_id: PaperId, reason: str) -> None:
        self.document_id = document_id
        self.reason = reason
        super().__init__(
            f"document {document_id} has an embedding for an unknown knowledge unit: {reason}"
        )


class EmbeddingStorageError(EmbeddingError):
    """Raised when a storage failure prevents embedding artifacts from being persisted."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(
            f"a storage error occurred while persisting embeddings for document {document_id}"
        )
