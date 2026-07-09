"""Integration tests for the real Neo4j provider.

Runs against a real Neo4j Community Edition instance (see
`docker-compose.yml` -- `docker compose up -d neo4j`), not a fake. Every
test uses a unique, randomly-generated document id and cleans up its own
nodes/relationships afterward, so tests don't interfere with each other or
leave state behind.
"""

from collections.abc import Iterator
from uuid import uuid4

import pytest

from backend.domain import PaperId
from backend.graph.models import GraphEdge, GraphNode, KnowledgeGraph, NodeLabel, RelationshipType
from backend.graph.providers.neo4j_provider import Neo4jProvider

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "kimrag-dev-password"


@pytest.fixture
def provider() -> Iterator[Neo4jProvider]:
    instance = Neo4jProvider(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    instance.ensure_schema()
    yield instance
    instance.close()


def _document_node(document_id: PaperId) -> GraphNode:
    return GraphNode(
        id=str(document_id),
        labels=(NodeLabel.DOCUMENT,),
        properties={"id": str(document_id), "paper_id": str(document_id)},
    )


def _knowledge_unit_node(document_id: PaperId, chunk_id: str) -> GraphNode:
    return GraphNode(
        id=chunk_id,
        labels=(NodeLabel.KNOWLEDGE_UNIT, NodeLabel.FIGURE_UNIT),
        properties={"id": chunk_id, "paper_id": str(document_id), "modality": "figure"},
    )


def _small_graph(document_id: PaperId) -> KnowledgeGraph:
    chunk_id = str(uuid4())
    return KnowledgeGraph(
        document_id=document_id,
        nodes=(_document_node(document_id), _knowledge_unit_node(document_id, chunk_id)),
        edges=(
            GraphEdge(
                source_id=chunk_id,
                target_id=str(document_id),
                relationship_type=RelationshipType.BELONGS_TO,
            ),
        ),
    )


def test_ensure_schema_is_idempotent(provider: Neo4jProvider) -> None:
    provider.ensure_schema()
    provider.ensure_schema()  # should not raise


def test_replace_graph_persists_nodes_and_relationships(provider: Neo4jProvider) -> None:
    document_id = PaperId(uuid4())
    graph = _small_graph(document_id)

    provider.replace_graph(graph)
    summary = provider.graph_summary(document_id)

    assert summary.node_count == 2
    assert summary.relationship_count == 1


def test_replace_graph_writes_correct_multi_labels(provider: Neo4jProvider) -> None:
    document_id = PaperId(uuid4())
    graph = _small_graph(document_id)
    provider.replace_graph(graph)

    with provider._driver.session(
        database=provider._database
    ) as session:  # test-only introspection
        labels = session.run(
            "MATCH (n:KnowledgeUnit {paper_id: $paper_id}) RETURN labels(n) AS labels",
            paper_id=str(document_id),
        ).single()["labels"]

    assert set(labels) == {"KGNode", "KnowledgeUnit", "FigureUnit"}


def test_replace_graph_is_idempotent(provider: Neo4jProvider) -> None:
    document_id = PaperId(uuid4())
    graph = _small_graph(document_id)

    provider.replace_graph(graph)
    provider.replace_graph(graph)  # re-run with the identical graph
    summary = provider.graph_summary(document_id)

    assert summary.node_count == 2
    assert summary.relationship_count == 1


def test_replace_graph_removes_stale_nodes_from_a_previous_run(provider: Neo4jProvider) -> None:
    document_id = PaperId(uuid4())
    first_graph = _small_graph(document_id)
    provider.replace_graph(first_graph)

    smaller_graph = KnowledgeGraph(
        document_id=document_id, nodes=(_document_node(document_id),), edges=()
    )
    provider.replace_graph(smaller_graph)
    summary = provider.graph_summary(document_id)

    assert summary.node_count == 1
    assert summary.relationship_count == 0


def test_graph_summary_is_zero_for_a_document_never_written(provider: Neo4jProvider) -> None:
    summary = provider.graph_summary(PaperId(uuid4()))

    assert summary.node_count == 0
    assert summary.relationship_count == 0


def test_replace_graph_does_not_affect_other_documents(provider: Neo4jProvider) -> None:
    document_a = PaperId(uuid4())
    document_b = PaperId(uuid4())
    provider.replace_graph(_small_graph(document_a))
    provider.replace_graph(_small_graph(document_b))

    provider.replace_graph(
        KnowledgeGraph(document_id=document_a, nodes=(_document_node(document_a),), edges=())
    )

    assert provider.graph_summary(document_a).node_count == 1
    assert provider.graph_summary(document_b).node_count == 2
