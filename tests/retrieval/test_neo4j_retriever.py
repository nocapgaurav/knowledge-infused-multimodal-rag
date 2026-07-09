"""Integration tests for the real Neo4j retriever.

Runs against a real Neo4j Community Edition instance (see
`docker-compose.yml` -- `docker compose up -d neo4j`), not a fake. Seeds
data via Module 8's own `Neo4jProvider` (the write side), then reads it
back through `Neo4jRetriever` (this module's read-only side).
"""

from collections.abc import Iterator
from uuid import uuid4

import pytest

from backend.domain import PaperId
from backend.graph.models import GraphEdge, GraphNode, KnowledgeGraph, NodeLabel
from backend.graph.models import RelationshipType as GraphRelationshipType
from backend.graph.providers.neo4j_provider import Neo4jProvider
from backend.retrieval.models import TraversalDirection
from backend.retrieval.providers.neo4j_retriever import Neo4jRetriever

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "kimrag-dev-password"


@pytest.fixture
def writer() -> Iterator[Neo4jProvider]:
    provider = Neo4jProvider(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    provider.ensure_schema()
    yield provider
    provider.close()


@pytest.fixture
def retriever() -> Iterator[Neo4jRetriever]:
    instance = Neo4jRetriever(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    yield instance
    instance.close()


def _node(node_id: str, document_id: PaperId, *labels: NodeLabel) -> GraphNode:
    return GraphNode(
        id=node_id, labels=labels, properties={"id": node_id, "paper_id": str(document_id)}
    )


def test_neighbors_outgoing_finds_the_next_chunk(
    writer: Neo4jProvider, retriever: Neo4jRetriever
) -> None:
    document_id = PaperId(uuid4())
    a, b = str(uuid4()), str(uuid4())
    graph = KnowledgeGraph(
        document_id=document_id,
        nodes=(
            _node(a, document_id, NodeLabel.KNOWLEDGE_UNIT, NodeLabel.TEXT_UNIT),
            _node(b, document_id, NodeLabel.KNOWLEDGE_UNIT, NodeLabel.TEXT_UNIT),
        ),
        edges=(GraphEdge(source_id=a, target_id=b, relationship_type=GraphRelationshipType.NEXT),),
    )
    writer.replace_graph(graph)

    neighbors = retriever.neighbors([a], ["NEXT"], TraversalDirection.OUTGOING)

    assert len(neighbors) == 1
    assert neighbors[0].source_id == a
    assert neighbors[0].neighbor_id == b
    assert set(neighbors[0].neighbor_labels) >= {"KnowledgeUnit", "TextUnit"}
    assert neighbors[0].relationship_type == "NEXT"


def test_neighbors_incoming_finds_the_predecessor(
    writer: Neo4jProvider, retriever: Neo4jRetriever
) -> None:
    document_id = PaperId(uuid4())
    a, b = str(uuid4()), str(uuid4())
    graph = KnowledgeGraph(
        document_id=document_id,
        nodes=(
            _node(a, document_id, NodeLabel.KNOWLEDGE_UNIT, NodeLabel.TEXT_UNIT),
            _node(b, document_id, NodeLabel.KNOWLEDGE_UNIT, NodeLabel.TEXT_UNIT),
        ),
        edges=(GraphEdge(source_id=a, target_id=b, relationship_type=GraphRelationshipType.NEXT),),
    )
    writer.replace_graph(graph)

    neighbors = retriever.neighbors([b], ["NEXT"], TraversalDirection.INCOMING)

    assert len(neighbors) == 1
    assert neighbors[0].source_id == b
    assert neighbors[0].neighbor_id == a


def test_neighbors_filters_by_relationship_type(
    writer: Neo4jProvider, retriever: Neo4jRetriever
) -> None:
    document_id = PaperId(uuid4())
    a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
    graph = KnowledgeGraph(
        document_id=document_id,
        nodes=(
            _node(a, document_id, NodeLabel.KNOWLEDGE_UNIT, NodeLabel.TEXT_UNIT),
            _node(b, document_id, NodeLabel.KNOWLEDGE_UNIT, NodeLabel.TEXT_UNIT),
            _node(c, document_id, NodeLabel.KNOWLEDGE_UNIT, NodeLabel.TEXT_UNIT),
        ),
        edges=(
            GraphEdge(source_id=a, target_id=b, relationship_type=GraphRelationshipType.NEXT),
            GraphEdge(source_id=a, target_id=c, relationship_type=GraphRelationshipType.CITES),
        ),
    )
    writer.replace_graph(graph)

    neighbors = retriever.neighbors([a], ["CITES"], TraversalDirection.OUTGOING)

    assert [n.neighbor_id for n in neighbors] == [c]


def test_neighbors_returns_section_labels_for_pass_through_nodes(
    writer: Neo4jProvider, retriever: Neo4jRetriever
) -> None:
    document_id = PaperId(uuid4())
    a, section = str(uuid4()), str(uuid4())
    graph = KnowledgeGraph(
        document_id=document_id,
        nodes=(
            _node(a, document_id, NodeLabel.KNOWLEDGE_UNIT, NodeLabel.TEXT_UNIT),
            _node(section, document_id, NodeLabel.SECTION),
        ),
        edges=(
            GraphEdge(
                source_id=a, target_id=section, relationship_type=GraphRelationshipType.BELONGS_TO
            ),
        ),
    )
    writer.replace_graph(graph)

    neighbors = retriever.neighbors([a], ["BELONGS_TO"], TraversalDirection.OUTGOING)

    assert neighbors[0].neighbor_labels == ("KGNode", "Section") or set(
        neighbors[0].neighbor_labels
    ) == {"KGNode", "Section"}


def test_neighbors_returns_empty_for_unknown_node() -> None:
    retriever = Neo4jRetriever(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    try:
        neighbors = retriever.neighbors([str(uuid4())], ["NEXT"], TraversalDirection.OUTGOING)
        assert neighbors == []
    finally:
        retriever.close()


def test_neighbors_with_empty_inputs_returns_empty_without_querying() -> None:
    retriever = Neo4jRetriever(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    try:
        assert retriever.neighbors([], ["NEXT"], TraversalDirection.OUTGOING) == []
        assert retriever.neighbors([str(uuid4())], [], TraversalDirection.OUTGOING) == []
    finally:
        retriever.close()
