"""Exceptions raised by the search index infrastructure."""

from backend.domain import PaperId


class SearchIndexError(Exception):
    """Base class for all search index infrastructure errors."""


class EmbeddingArtifactsNotFoundError(SearchIndexError):
    """Raised when no embedding artifacts exist for a given document id."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"no embedding artifacts found for {document_id}")


class VectorStoreError(SearchIndexError):
    """Raised by a VectorStore provider when an operation fails."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"vector store operation failed: {reason}")


class NoVectorsIndexedError(SearchIndexError):
    """Raised when every vector failed to index."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"document {document_id} had no vectors successfully indexed")


class MultiCollectionIndexingNotSupportedError(SearchIndexError):
    """Raised when a document's embeddings span more than one collection.

    Reachable only once a concrete image embedding provider exists (Module
    6 does not implement one yet) -- the single-collection `IndexManifest`
    shape does not yet model multiple collections per document. Failing
    loudly here is preferable to silently indexing only one collection and
    losing the others.
    """

    def __init__(self, *, document_id: PaperId, collection_names: list[str]) -> None:
        self.document_id = document_id
        self.collection_names = collection_names
        super().__init__(
            f"document {document_id} produced embeddings spanning multiple "
            f"collections {collection_names}, which is not yet supported"
        )


class IndexValidationError(SearchIndexError):
    """Base class for post-indexing verification failures."""


class CollectionMissingError(IndexValidationError):
    """Raised when verification finds the expected collection does not exist."""

    def __init__(self, *, document_id: PaperId, collection: str) -> None:
        self.document_id = document_id
        self.collection = collection
        super().__init__(f"collection '{collection}' does not exist for document {document_id}")


class DimensionMismatchError(IndexValidationError):
    """Raised when a collection's actual vector dimension doesn't match expectations."""

    def __init__(self, *, document_id: PaperId, expected: int, actual: int) -> None:
        self.document_id = document_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"document {document_id}: expected dimension {expected}, collection has {actual}"
        )


class IndexedCountMismatchError(IndexValidationError):
    """Raised when the indexed point count doesn't match the expected count."""

    def __init__(self, *, document_id: PaperId, expected: int, actual: int) -> None:
        self.document_id = document_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"document {document_id}: expected {expected} indexed vectors, found {actual}"
        )


class PayloadIntegrityError(IndexValidationError):
    """Raised when a sampled point's payload fails integrity checks."""

    def __init__(self, *, document_id: PaperId, reason: str) -> None:
        self.document_id = document_id
        self.reason = reason
        super().__init__(f"document {document_id}: payload integrity check failed: {reason}")


class IndexStorageError(SearchIndexError):
    """Raised when a storage failure prevents index artifacts from being persisted."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(
            f"a storage error occurred while persisting the index for document {document_id}"
        )
