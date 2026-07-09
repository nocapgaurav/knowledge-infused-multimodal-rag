"""Neo4j-based implementation of `GraphRetriever`.

This is the only file in this module permitted to import `neo4j`. Owns its
own driver connection rather than wrapping Module 8's `Neo4jProvider` --
composing over a write-capable object would let a future change reach its
`replace_graph` method through this module, defeating the point of a
narrower, read-only interface.

Confirmed against a real Neo4j 5 Community Edition instance: `type(r) IN
$list` and `x.id IN $list` are both ordinary parameterized Cypher (no
label-parameterization issue here, since this file only ever reads
`labels(y)`, never writes a label).
"""

from collections.abc import Sequence

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from backend.retrieval.exceptions import GraphRetrieverError
from backend.retrieval.interfaces.graph_retriever import GraphRetriever
from backend.retrieval.models import GraphNeighbor, TraversalDirection

_COMMON_NODE_LABEL = "KGNode"
"""Matches the technical label every node carries, written by Module 8's
`Neo4jProvider` -- scopes every query to knowledge-graph nodes only."""


class Neo4jRetriever(GraphRetriever):
    """Read-only graph traversal backed by a Neo4j instance."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        """Connect to a Neo4j instance.

        Args:
            uri: Bolt URI (e.g. "bolt://localhost:7687").
            user: Username to authenticate with.
            password: Password to authenticate with.
            database: Name of the database within the instance to use.
        """
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        """Release the underlying driver's connection pool."""
        self._driver.close()

    def neighbors(
        self,
        node_ids: Sequence[str],
        relationship_types: Sequence[str],
        direction: TraversalDirection,
    ) -> list[GraphNeighbor]:
        if not node_ids or not relationship_types:
            return []
        if direction is TraversalDirection.OUTGOING:
            query = (
                f"MATCH (x:{_COMMON_NODE_LABEL})-[r]->(y:{_COMMON_NODE_LABEL}) "
                f"WHERE x.id IN $node_ids AND type(r) IN $relationship_types "
                f"RETURN x.id AS source_id, y.id AS neighbor_id, labels(y) AS neighbor_labels, "
                f"type(r) AS relationship_type"
            )
        else:
            query = (
                f"MATCH (x:{_COMMON_NODE_LABEL})<-[r]-(y:{_COMMON_NODE_LABEL}) "
                f"WHERE x.id IN $node_ids AND type(r) IN $relationship_types "
                f"RETURN x.id AS source_id, y.id AS neighbor_id, labels(y) AS neighbor_labels, "
                f"type(r) AS relationship_type"
            )
        try:
            with self._driver.session(database=self._database) as session:
                records = list(
                    session.run(
                        query,
                        node_ids=list(node_ids),
                        relationship_types=list(relationship_types),
                    )
                )
        except Neo4jError as exc:
            raise GraphRetrieverError(reason=str(exc)) from exc

        return [
            GraphNeighbor(
                source_id=record["source_id"],
                neighbor_id=record["neighbor_id"],
                neighbor_labels=tuple(record["neighbor_labels"]),
                relationship_type=record["relationship_type"],
                direction=direction,
            )
            for record in records
        ]
