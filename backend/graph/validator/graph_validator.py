"""Pre-persistence structural verification of a freshly constructed graph.

Distinct from post-write verification (checked *after* a store call, to
confirm it took effect): this runs *before* any store is involved, against
the in-memory `KnowledgeGraph` itself, so a defect is caught before it can
ever be written. Every check here is a genuine construction-bug detector,
not a defensive check against input this module's own builders could
never produce -- see each check's docstring for why it's reachable at all.
"""

import logging

from backend.domain import Chunk, PaperId, Relationship
from backend.graph.exceptions import (
    DanglingEdgeError,
    DuplicateEdgeError,
    DuplicateNodeError,
    GraphCompletenessError,
    InvalidEdgeEndpointTypeError,
    OrphanNodeError,
)
from backend.graph.models import GraphEdge, GraphNode, KnowledgeGraph, NodeLabel, RelationshipType

logger = logging.getLogger(__name__)

_KNOWLEDGE_UNIT_ONLY_RELATIONSHIPS = frozenset(
    {
        RelationshipType.NEXT,
        RelationshipType.CITES,
        RelationshipType.REFERENCES,
        RelationshipType.CONTINUES,
    }
)


class GraphValidator:
    """Verifies a freshly constructed `KnowledgeGraph` before it may be persisted."""

    def validate(
        self,
        document_id: PaperId,
        graph: KnowledgeGraph,
        chunks: list[Chunk],
        relationships: list[Relationship],
    ) -> None:
        """Verify a graph is internally consistent and faithful to its source.

        Args:
            document_id: Identifier of the document the graph represents.
            graph: The freshly constructed graph to verify.
            chunks: The knowledge units the graph was built from, for the
                completeness check.
            relationships: The relationships the graph was built from, for
                the completeness check.

        Raises:
            DuplicateNodeError: The same node id appears more than once.
            DuplicateEdgeError: The same (source, target, type) triple
                appears more than once.
            DanglingEdgeError: An edge references a node id absent from the graph.
            InvalidEdgeEndpointTypeError: An edge connects node labels its
                type does not permit.
            OrphanNodeError: A non-root node has no edges at all.
            GraphCompletenessError: The graph does not account for every
                source chunk or relationship.
        """
        node_by_id = self._check_no_duplicate_nodes(document_id, graph.nodes)
        self._check_no_duplicate_edges(document_id, graph.edges)
        self._check_no_dangling_edges(document_id, graph.edges, node_by_id)
        self._check_edge_endpoint_types(document_id, graph.edges, node_by_id)
        self._check_no_orphan_nodes(document_id, graph.nodes, graph.edges)
        self._check_completeness(document_id, graph, chunks, relationships)
        logger.info(
            "graph validated",
            extra={
                "document_id": str(document_id),
                "nodes": graph.node_count,
                "relationships": graph.relationship_count,
            },
        )

    def _check_no_duplicate_nodes(
        self, document_id: PaperId, nodes: tuple[GraphNode, ...]
    ) -> dict[str, GraphNode]:
        # Unreachable via GraphPlanner today (every node id is a distinct
        # chunk/section/document id), but fails loudly rather than silently
        # dropping data if a future builder change ever produces a collision.
        node_by_id: dict[str, GraphNode] = {}
        for node in nodes:
            if node.id in node_by_id:
                raise DuplicateNodeError(document_id=document_id, node_id=node.id)
            node_by_id[node.id] = node
        return node_by_id

    def _check_no_duplicate_edges(self, document_id: PaperId, edges: tuple[GraphEdge, ...]) -> None:
        seen: set[tuple[str, str, RelationshipType]] = set()
        for edge in edges:
            key = edge.identity_key
            if key in seen:
                raise DuplicateEdgeError(
                    document_id=document_id,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    relationship_type=edge.relationship_type,
                )
            seen.add(key)

    def _check_no_dangling_edges(
        self,
        document_id: PaperId,
        edges: tuple[GraphEdge, ...],
        node_by_id: dict[str, GraphNode],
    ) -> None:
        for edge in edges:
            for endpoint, role in ((edge.source_id, "source"), (edge.target_id, "target")):
                if endpoint not in node_by_id:
                    raise DanglingEdgeError(
                        document_id=document_id,
                        edge_description=(
                            f"({edge.source_id})-[{edge.relationship_type}]->({edge.target_id})"
                        ),
                        missing_id=f"{role}={endpoint}",
                    )

    def _check_edge_endpoint_types(
        self,
        document_id: PaperId,
        edges: tuple[GraphEdge, ...],
        node_by_id: dict[str, GraphNode],
    ) -> None:
        for edge in edges:
            if edge.relationship_type in _KNOWLEDGE_UNIT_ONLY_RELATIONSHIPS:
                for endpoint, role in ((edge.source_id, "source"), (edge.target_id, "target")):
                    if NodeLabel.KNOWLEDGE_UNIT not in node_by_id[endpoint].labels:
                        raise InvalidEdgeEndpointTypeError(
                            document_id=document_id,
                            reason=(
                                f"{edge.relationship_type} {role} {endpoint} is not a "
                                f"KnowledgeUnit (labels={node_by_id[endpoint].labels})"
                            ),
                        )
            elif (
                edge.relationship_type is RelationshipType.BELONGS_TO
                and NodeLabel.DOCUMENT in node_by_id[edge.source_id].labels
            ):
                raise InvalidEdgeEndpointTypeError(
                    document_id=document_id,
                    reason=f"BELONGS_TO source {edge.source_id} must not be a Document",
                )

    def _check_no_orphan_nodes(
        self,
        document_id: PaperId,
        nodes: tuple[GraphNode, ...],
        edges: tuple[GraphEdge, ...],
    ) -> None:
        # Unreachable via GraphPlanner today (every Section/KnowledgeUnit
        # node always gets a BELONGS_TO edge by construction); fails loudly
        # if a future builder change ever produces an unconnected node.
        connected: set[str] = set()
        for edge in edges:
            connected.add(edge.source_id)
            connected.add(edge.target_id)
        for node in nodes:
            if NodeLabel.DOCUMENT in node.labels:
                continue
            if node.id not in connected:
                raise OrphanNodeError(document_id=document_id, node_id=node.id)

    def _check_completeness(
        self,
        document_id: PaperId,
        graph: KnowledgeGraph,
        chunks: list[Chunk],
        relationships: list[Relationship],
    ) -> None:
        knowledge_unit_ids = {
            node.id for node in graph.nodes if NodeLabel.KNOWLEDGE_UNIT in node.labels
        }
        expected_chunk_ids = {str(chunk.id) for chunk in chunks}
        if knowledge_unit_ids != expected_chunk_ids:
            raise GraphCompletenessError(
                document_id=document_id,
                reason=(
                    f"expected {len(expected_chunk_ids)} knowledge unit nodes, "
                    f"found {len(knowledge_unit_ids)}"
                ),
            )

        # Every domain Relationship has a 1:1 graph edge type (see
        # relationship_builder._DOMAIN_TO_GRAPH_RELATIONSHIP), so the
        # expected count is simply the number of source relationships.
        expected_projected_edges = len(relationships)
        actual_projected_edges = sum(
            1
            for edge in graph.edges
            if edge.relationship_type
            in (RelationshipType.CITES, RelationshipType.REFERENCES, RelationshipType.CONTINUES)
        )
        if actual_projected_edges != expected_projected_edges:
            raise GraphCompletenessError(
                document_id=document_id,
                reason=(
                    f"expected {expected_projected_edges} projected relationship edges, "
                    f"found {actual_projected_edges}"
                ),
            )
