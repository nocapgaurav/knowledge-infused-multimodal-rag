"""Exceptions raised by the knowledge graph infrastructure."""

from backend.domain import PaperId
from backend.graph.models.graph_edge import RelationshipType


class GraphError(Exception):
    """Base class for all knowledge graph infrastructure errors."""


class KnowledgeRepresentationNotFoundError(GraphError):
    """Raised when no knowledge representation exists for a given document id."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"no knowledge representation found for {document_id}")


class GraphValidationError(GraphError):
    """Base class for structural defects found in a freshly constructed graph.

    Every subclass is raised *before* persistence -- these are checks
    against the in-memory `KnowledgeGraph`, not the database.
    """


class DuplicateNodeError(GraphValidationError):
    """Raised when the same node id appears more than once in a graph."""

    def __init__(self, *, document_id: PaperId, node_id: str) -> None:
        self.document_id = document_id
        self.node_id = node_id
        super().__init__(f"document {document_id} has a duplicate graph node: {node_id}")


class DuplicateEdgeError(GraphValidationError):
    """Raised when the same (source, target, type) triple appears more than once."""

    def __init__(
        self,
        *,
        document_id: PaperId,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
    ) -> None:
        self.document_id = document_id
        self.source_id = source_id
        self.target_id = target_id
        self.relationship_type = relationship_type
        super().__init__(
            f"document {document_id} has a duplicate edge: "
            f"({source_id})-[{relationship_type}]->({target_id})"
        )


class OrphanNodeError(GraphValidationError):
    """Raised when a non-root node has no edges connecting it to the rest of the graph."""

    def __init__(self, *, document_id: PaperId, node_id: str) -> None:
        self.document_id = document_id
        self.node_id = node_id
        super().__init__(f"document {document_id} has an orphan graph node: {node_id}")


class DanglingEdgeError(GraphValidationError):
    """Raised when an edge references a node id that does not exist in the graph."""

    def __init__(self, *, document_id: PaperId, edge_description: str, missing_id: str) -> None:
        self.document_id = document_id
        self.edge_description = edge_description
        self.missing_id = missing_id
        super().__init__(
            f"document {document_id} has a dangling edge {edge_description}: "
            f"node {missing_id} does not exist"
        )


class InvalidEdgeEndpointTypeError(GraphValidationError):
    """Raised when an edge connects node labels its relationship type does not permit."""

    def __init__(self, *, document_id: PaperId, reason: str) -> None:
        self.document_id = document_id
        self.reason = reason
        super().__init__(f"document {document_id} has an edge with an invalid endpoint: {reason}")


class GraphCompletenessError(GraphValidationError):
    """Raised when the graph does not fully account for its source chunks/relationships."""

    def __init__(self, *, document_id: PaperId, reason: str) -> None:
        self.document_id = document_id
        self.reason = reason
        super().__init__(f"document {document_id} graph is incomplete: {reason}")


class GraphStoreError(GraphError):
    """Raised by a KnowledgeGraphStore provider when an operation fails."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"graph store operation failed: {reason}")


class GraphStorageError(GraphError):
    """Raised when a storage failure prevents a graph manifest from being persisted."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(
            f"a storage error occurred while persisting the graph for document {document_id}"
        )
