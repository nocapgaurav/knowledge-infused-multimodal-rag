"""GraphRetriever: the read-only port evidence expansion depends on.

Deliberately narrower than Module 8's `KnowledgeGraphStore`: this module
must never write to Neo4j, and depending on the write-capable interface
(`replace_graph`) would make read-only a matter of convention rather than
a structural guarantee. The single method here reports one hop of
neighbors; `GraphExpander` is the only place multi-hop traversal,
budgeting, cycle prevention, and deduplication are decided -- the provider
never makes those decisions itself.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from backend.retrieval.models import GraphNeighbor, TraversalDirection


class GraphRetriever(ABC):
    """Read-only single-hop neighbor lookup against a graph database."""

    @abstractmethod
    def neighbors(
        self,
        node_ids: Sequence[str],
        relationship_types: Sequence[str],
        direction: TraversalDirection,
    ) -> list[GraphNeighbor]:
        """Return the immediate neighbors of the given nodes via the given relationship types.

        Args:
            node_ids: Ids of the nodes to find neighbors of.
            relationship_types: Relationship type names to follow (e.g.
                `["NEXT", "CITES"]`). A node connected only by a
                relationship type not in this list is not returned.
            direction: Which direction of each relationship type to follow.

        Returns:
            Every matching neighbor found, including duplicates across
            different `node_ids` -- deduplication is the caller's
            responsibility, since only the caller knows the traversal's
            global visited set.

        Raises:
            GraphRetrieverError: The lookup failed.
        """
