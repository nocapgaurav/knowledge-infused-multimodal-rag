"""Neo4j-based implementation of `KnowledgeGraphStore`.

This is the only file in the application permitted to import `neo4j`.

Cypher does not accept a node's label set as a query parameter (confirmed
against a real Neo4j 5 Community Edition instance -- it raises a
`CypherSyntaxError`), so labels must be written literally into each query.
That's safe here only because every label written is drawn from this
module's own closed `NodeLabel`/`RelationshipType` enums, never from
external input -- there is no Cypher-injection surface.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence

from neo4j import GraphDatabase, ManagedTransaction, Result
from neo4j.exceptions import Neo4jError

from backend.domain import PaperId
from backend.graph.exceptions import GraphStoreError
from backend.graph.interfaces.knowledge_graph_store import KnowledgeGraphStore
from backend.graph.models import GraphEdge, GraphNode, GraphSummary, KnowledgeGraph

logger = logging.getLogger(__name__)

_COMMON_NODE_LABEL = "KGNode"
"""Technical label every node this provider writes carries, in addition to
its domain labels -- gives the uniqueness constraint and the
delete/summary queries below a single, vendor-level anchor to scope on,
independent of which domain labels (`Document`, `Section`, `KnowledgeUnit`
+ modality) a given node has."""


class Neo4jProvider(KnowledgeGraphStore):
    """Knowledge graph store backed by a Neo4j instance."""

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

    def ensure_schema(self) -> None:
        try:
            with self._driver.session(database=self._database) as session:
                session.run(
                    f"CREATE CONSTRAINT graph_node_id_unique IF NOT EXISTS "
                    f"FOR (n:{_COMMON_NODE_LABEL}) REQUIRE n.id IS UNIQUE"
                )
        except Neo4jError as exc:
            raise GraphStoreError(reason=str(exc)) from exc

    def replace_graph(self, graph: KnowledgeGraph) -> None:
        try:
            with self._driver.session(database=self._database) as session:
                session.execute_write(self._replace_graph_tx, graph)
        except Neo4jError as exc:
            raise GraphStoreError(reason=str(exc)) from exc
        logger.info(
            "graph replaced",
            extra={
                "document_id": str(graph.document_id),
                "nodes": graph.node_count,
                "relationships": graph.relationship_count,
            },
        )

    def graph_summary(self, document_id: PaperId) -> GraphSummary:
        try:
            with self._driver.session(database=self._database) as session:
                node_count = _scalar_count(
                    session.run(
                        f"MATCH (n:{_COMMON_NODE_LABEL} {{paper_id: $paper_id}}) "
                        f"RETURN count(n) AS c",
                        paper_id=str(document_id),
                    )
                )
                relationship_count = _scalar_count(
                    session.run(
                        f"MATCH (a:{_COMMON_NODE_LABEL} {{paper_id: $paper_id}})-[r]->() "
                        f"RETURN count(r) AS c",
                        paper_id=str(document_id),
                    )
                )
        except Neo4jError as exc:
            raise GraphStoreError(reason=str(exc)) from exc
        return GraphSummary(node_count=node_count, relationship_count=relationship_count)

    def _replace_graph_tx(self, tx: ManagedTransaction, graph: KnowledgeGraph) -> None:
        tx.run(
            f"MATCH (n:{_COMMON_NODE_LABEL} {{paper_id: $paper_id}}) DETACH DELETE n",
            paper_id=str(graph.document_id),
        )
        for label_clause, nodes in _group_nodes_by_label_clause(graph.nodes).items():
            tx.run(
                f"UNWIND $rows AS row "
                f"MERGE (n:{label_clause} {{id: row.id}}) "
                f"SET n += row.properties",
                rows=[{"id": node.id, "properties": node.properties} for node in nodes],
            )
        for relationship_type, edges in _group_edges_by_type(graph.edges).items():
            tx.run(
                f"UNWIND $rows AS row "
                f"MATCH (a:{_COMMON_NODE_LABEL} {{id: row.source_id}}), "
                f"(b:{_COMMON_NODE_LABEL} {{id: row.target_id}}) "
                f"MERGE (a)-[r:{relationship_type}]->(b) "
                f"SET r += row.properties",
                rows=[
                    {
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "properties": edge.properties,
                    }
                    for edge in edges
                ],
            )


def _group_nodes_by_label_clause(nodes: Sequence[GraphNode]) -> dict[str, list[GraphNode]]:
    groups: dict[str, list[GraphNode]] = defaultdict(list)
    for node in nodes:
        label_clause = ":".join([_COMMON_NODE_LABEL, *(label.value for label in node.labels)])
        groups[label_clause].append(node)
    return groups


def _group_edges_by_type(edges: Sequence[GraphEdge]) -> dict[str, list[GraphEdge]]:
    groups: dict[str, list[GraphEdge]] = defaultdict(list)
    for edge in edges:
        groups[edge.relationship_type.value].append(edge)
    return groups


def _scalar_count(result: Result) -> int:
    record = result.single()
    assert record is not None, "a count() aggregate query always returns exactly one row"
    return int(record["c"])
