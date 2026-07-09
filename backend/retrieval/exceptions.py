"""Exceptions raised by the hybrid evidence retrieval engine."""

from backend.domain import PaperId


class RetrievalError(Exception):
    """Base class for all retrieval engine errors."""


class DocumentNotIndexedError(RetrievalError):
    """Raised when no index manifest exists for a given document id.

    Means Module 7 has never indexed this document -- there is no
    collection to search.
    """

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"document {document_id} has not been indexed")


class DocumentNotGraphedError(RetrievalError):
    """Raised when no graph manifest exists for a given document id.

    Means Module 8 has never built a graph for this document -- there is
    nothing to expand through.
    """

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"document {document_id} has no knowledge graph")


class QueryEmbeddingError(RetrievalError):
    """Raised when the query text could not be embedded."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"failed to embed query: {reason}")


class VectorRetrieverError(RetrievalError):
    """Raised by a VectorRetriever provider when an operation fails."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"vector retriever operation failed: {reason}")


class GraphRetrieverError(RetrievalError):
    """Raised by a GraphRetriever provider when an operation fails."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"graph retriever operation failed: {reason}")


class RetrievalValidationError(RetrievalError):
    """Base class for structural defects found during retrieval."""


class DuplicateCandidateError(RetrievalValidationError):
    """Raised when the same knowledge unit appears more than once in the candidate pool."""

    def __init__(self, *, knowledge_unit_id: str) -> None:
        self.knowledge_unit_id = knowledge_unit_id
        super().__init__(f"duplicate candidate: {knowledge_unit_id}")


class GraphCycleError(RetrievalValidationError):
    """Raised when a traversal path revisits a node it already passed through."""

    def __init__(self, *, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(f"traversal path revisits node {node_id}, forming a cycle")


class DuplicateEvidenceError(RetrievalValidationError):
    """Raised when the same candidate appears in more than one evidence group."""

    def __init__(self, *, knowledge_unit_id: str) -> None:
        self.knowledge_unit_id = knowledge_unit_id
        super().__init__(f"candidate {knowledge_unit_id} assigned to more than one evidence group")


class MissingKnowledgeUnitError(RetrievalValidationError):
    """Raised when an evidence group references a candidate absent from the bundle's pool."""

    def __init__(self, *, knowledge_unit_id: str) -> None:
        self.knowledge_unit_id = knowledge_unit_id
        super().__init__(f"evidence group references unknown candidate {knowledge_unit_id}")


class RankingConsistencyError(RetrievalValidationError):
    """Raised when ranking explanations are internally inconsistent."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"ranking is inconsistent: {reason}")


class BundleConsistencyError(RetrievalValidationError):
    """Raised when the assembled evidence bundle fails a whole-bundle consistency check."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"evidence bundle is inconsistent: {reason}")


class TraceCompletenessError(RetrievalValidationError):
    """Raised when the retrieval trace does not cover every expected phase."""

    def __init__(self, *, missing_phases: list[str]) -> None:
        self.missing_phases = missing_phases
        super().__init__(f"retrieval trace is missing phases: {missing_phases}")


class RetrievalStorageError(RetrievalError):
    """Raised when a storage failure prevents a retrieval manifest from being persisted."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(
            f"a storage error occurred while persisting the retrieval manifest for "
            f"document {document_id}"
        )
