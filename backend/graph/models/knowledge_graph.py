"""KnowledgeGraph: the complete, in-memory graph for one document.

Constructed by `GraphPlanner` from Module 5's artifacts, checked by
`GraphValidator`, and only then handed to a `KnowledgeGraphStore` for
persistence. Nothing in this module ever builds a Neo4j node or
relationship directly from `Chunk`/`Relationship` -- it always passes
through this intermediate, database-independent representation first.
"""

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import PaperId
from backend.graph.models.graph_edge import GraphEdge
from backend.graph.models.graph_node import GraphNode


class KnowledgeGraph(BaseModel):
    """The complete set of nodes and edges derived from one document.

    Attributes:
        document_id: Identifier of the document this graph represents.
        nodes: Every node in the graph.
        edges: Every edge in the graph.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: PaperId
    nodes: tuple[GraphNode, ...] = Field(default_factory=tuple)
    edges: tuple[GraphEdge, ...] = Field(default_factory=tuple)

    @property
    def node_count(self) -> int:
        """Number of nodes in this graph."""
        return len(self.nodes)

    @property
    def relationship_count(self) -> int:
        """Number of edges in this graph."""
        return len(self.edges)
