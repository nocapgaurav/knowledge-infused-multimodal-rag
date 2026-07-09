"""KnowledgeGraphStore: the port every concrete graph database backend implements.

Business logic (planner, service, validator) depends only on this
interface, never on a concrete backend. Swapping Neo4j for ArangoDB,
Neptune, or JanusGraph later means writing one new provider file -- zero
changes anywhere else, including in Module 9 (hybrid retrieval), which will
depend on this same interface for traversal reads.
"""

from abc import ABC, abstractmethod

from backend.domain import PaperId
from backend.graph.models import GraphSummary, KnowledgeGraph


class KnowledgeGraphStore(ABC):
    """A graph database capable of storing a document's knowledge graph."""

    @abstractmethod
    def ensure_schema(self) -> None:
        """Create any schema-level constraints the store needs, if absent.

        Idempotent: calling this repeatedly is safe.

        Raises:
            GraphStoreError: The schema could not be ensured.
        """

    @abstractmethod
    def replace_graph(self, graph: KnowledgeGraph) -> None:
        """Replace a document's entire stored graph with the given one.

        Idempotent and atomic: any existing nodes/edges for
        `graph.document_id` are removed and the given graph is written in
        their place, as a single operation -- never merged incrementally
        with a previous run's now-stale nodes or edges. Only a graph that
        has already passed `GraphValidator` should ever reach this call.

        Args:
            graph: The validated graph to persist.

        Raises:
            GraphStoreError: The replacement failed.
        """

    @abstractmethod
    def graph_summary(self, document_id: PaperId) -> GraphSummary:
        """Return the node and edge counts currently stored for a document.

        Args:
            document_id: Identifier of the document to summarize.

        Returns:
            The document's current `GraphSummary`. Both counts are `0` if
            no graph has ever been stored for this document.

        Raises:
            GraphStoreError: The summary could not be read.
        """
