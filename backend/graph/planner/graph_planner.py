"""Plans (constructs) a complete KnowledgeGraph from a document's knowledge representation."""

from collections.abc import Sequence

from backend.domain import Chunk, PaperId, Relationship
from backend.graph.builders.node_builder import (
    build_document_node,
    build_knowledge_unit_nodes,
    build_section_nodes,
)
from backend.graph.builders.relationship_builder import (
    build_containment_edges,
    build_relationship_edges,
    build_sequence_edges,
)
from backend.graph.models import KnowledgeGraph


class GraphPlanner:
    """Builds a complete, database-independent `KnowledgeGraph` for one document.

    Stateless: every node and edge is derived purely from its arguments, so
    calling `plan` twice with the same inputs always produces an identical graph.
    """

    def plan(
        self,
        document_id: PaperId,
        chunks: Sequence[Chunk],
        relationships: Sequence[Relationship],
    ) -> KnowledgeGraph:
        """Construct the graph for a document.

        Args:
            document_id: Identifier of the document.
            chunks: The document's knowledge units (Module 5's output).
            relationships: The document's relationships (Module 5's output).

        Returns:
            The document's complete, unvalidated `KnowledgeGraph`. Callers
            must run it through `GraphValidator` before persisting it.
        """
        nodes = [
            build_document_node(document_id),
            *build_section_nodes(document_id, chunks),
            *build_knowledge_unit_nodes(chunks),
        ]
        edges = [
            *build_containment_edges(document_id, chunks),
            *build_sequence_edges(chunks),
            *build_relationship_edges(relationships),
        ]
        return KnowledgeGraph(document_id=document_id, nodes=tuple(nodes), edges=tuple(edges))
