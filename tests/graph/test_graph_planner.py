"""Tests for GraphPlanner: end-to-end construction of a complete KnowledgeGraph."""

from uuid import uuid4

from backend.domain import Chunk, ChunkModality, PaperId, Relationship, SectionId
from backend.domain import RelationshipType as DomainRelationshipType
from backend.graph.models import NodeLabel, RelationshipType
from backend.graph.planner.graph_planner import GraphPlanner


def _chunk(
    paper_id: PaperId,
    order: int,
    section_id: SectionId | None = None,
    modality: ChunkModality = ChunkModality.TEXT,
) -> Chunk:
    return Chunk(
        paper_id=paper_id, section_id=section_id, order=order, modality=modality, text="text"
    )


def test_plan_produces_one_document_node() -> None:
    document_id = PaperId(uuid4())
    graph = GraphPlanner().plan(document_id, [_chunk(document_id, 0)], [])

    document_nodes = [node for node in graph.nodes if NodeLabel.DOCUMENT in node.labels]
    assert len(document_nodes) == 1
    assert document_nodes[0].id == str(document_id)


def test_plan_produces_one_knowledge_unit_node_per_chunk() -> None:
    document_id = PaperId(uuid4())
    chunks = [_chunk(document_id, i) for i in range(3)]

    graph = GraphPlanner().plan(document_id, chunks, [])

    knowledge_unit_nodes = [node for node in graph.nodes if NodeLabel.KNOWLEDGE_UNIT in node.labels]
    assert {node.id for node in knowledge_unit_nodes} == {str(chunk.id) for chunk in chunks}


def test_plan_produces_section_nodes_and_relationship_edges() -> None:
    document_id = PaperId(uuid4())
    section_id = SectionId(uuid4())
    chunk_a = _chunk(document_id, 0, section_id=section_id)
    chunk_b = _chunk(document_id, 1)
    relationship = Relationship(
        paper_id=document_id,
        source_chunk_id=chunk_b.id,
        target_chunk_id=chunk_a.id,
        relationship_type=DomainRelationshipType.CITES,
    )

    graph = GraphPlanner().plan(document_id, [chunk_a, chunk_b], [relationship])

    section_nodes = [node for node in graph.nodes if NodeLabel.SECTION in node.labels]
    assert len(section_nodes) == 1
    cites_edges = [edge for edge in graph.edges if edge.relationship_type is RelationshipType.CITES]
    assert len(cites_edges) == 1
    assert cites_edges[0].source_id == str(chunk_b.id)
    assert cites_edges[0].target_id == str(chunk_a.id)


def test_plan_node_and_edge_counts_are_deterministic_across_repeated_calls() -> None:
    document_id = PaperId(uuid4())
    chunks = [_chunk(document_id, i) for i in range(4)]
    planner = GraphPlanner()

    first = planner.plan(document_id, chunks, [])
    second = planner.plan(document_id, chunks, [])

    assert first.nodes == second.nodes
    assert first.edges == second.edges
