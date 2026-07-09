"""Tests for GraphValidator's pre-persistence structural checks.

Most defects here are unreachable through `GraphPlanner` (its output is
correct by construction) -- these tests exercise the validator directly
against hand-crafted, deliberately malformed graphs, proving each check
actually fires if a future builder change ever introduces the defect.
"""

from uuid import uuid4

import pytest

from backend.domain import Chunk, ChunkModality, PaperId
from backend.graph.exceptions import (
    DanglingEdgeError,
    DuplicateEdgeError,
    DuplicateNodeError,
    GraphCompletenessError,
    InvalidEdgeEndpointTypeError,
    OrphanNodeError,
)
from backend.graph.models import GraphEdge, GraphNode, KnowledgeGraph, NodeLabel, RelationshipType
from backend.graph.validator.graph_validator import GraphValidator


def _chunk(paper_id: PaperId, order: int = 0) -> Chunk:
    return Chunk(paper_id=paper_id, order=order, modality=ChunkModality.TEXT, text="text")


def _document_node(document_id: PaperId) -> GraphNode:
    return GraphNode(
        id=str(document_id),
        labels=(NodeLabel.DOCUMENT,),
        properties={"id": str(document_id)},
    )


def _knowledge_unit_node(chunk: Chunk) -> GraphNode:
    return GraphNode(
        id=str(chunk.id),
        labels=(NodeLabel.KNOWLEDGE_UNIT, NodeLabel.TEXT_UNIT),
        properties={"id": str(chunk.id)},
    )


def test_valid_graph_passes() -> None:
    document_id = PaperId(uuid4())
    chunk = _chunk(document_id)
    graph = KnowledgeGraph(
        document_id=document_id,
        nodes=(_document_node(document_id), _knowledge_unit_node(chunk)),
        edges=(
            GraphEdge(
                source_id=str(chunk.id),
                target_id=str(document_id),
                relationship_type=RelationshipType.BELONGS_TO,
            ),
        ),
    )

    GraphValidator().validate(document_id, graph, [chunk], [])  # should not raise


def test_duplicate_node_raises() -> None:
    document_id = PaperId(uuid4())
    chunk = _chunk(document_id)
    node = _knowledge_unit_node(chunk)
    graph = KnowledgeGraph(
        document_id=document_id,
        nodes=(_document_node(document_id), node, node),
        edges=(),
    )

    with pytest.raises(DuplicateNodeError):
        GraphValidator().validate(document_id, graph, [chunk], [])


def test_duplicate_edge_raises() -> None:
    document_id = PaperId(uuid4())
    chunk = _chunk(document_id)
    edge = GraphEdge(
        source_id=str(chunk.id),
        target_id=str(document_id),
        relationship_type=RelationshipType.BELONGS_TO,
    )
    graph = KnowledgeGraph(
        document_id=document_id,
        nodes=(_document_node(document_id), _knowledge_unit_node(chunk)),
        edges=(edge, edge),
    )

    with pytest.raises(DuplicateEdgeError):
        GraphValidator().validate(document_id, graph, [chunk], [])


def test_dangling_edge_raises() -> None:
    document_id = PaperId(uuid4())
    chunk = _chunk(document_id)
    graph = KnowledgeGraph(
        document_id=document_id,
        nodes=(_document_node(document_id), _knowledge_unit_node(chunk)),
        edges=(
            GraphEdge(
                source_id=str(chunk.id),
                target_id=str(uuid4()),  # does not exist in the graph
                relationship_type=RelationshipType.BELONGS_TO,
            ),
        ),
    )

    with pytest.raises(DanglingEdgeError):
        GraphValidator().validate(document_id, graph, [chunk], [])


def test_orphan_node_raises() -> None:
    document_id = PaperId(uuid4())
    chunk = _chunk(document_id)
    other_chunk = _chunk(document_id, order=1)
    graph = KnowledgeGraph(
        document_id=document_id,
        nodes=(
            _document_node(document_id),
            _knowledge_unit_node(chunk),
            _knowledge_unit_node(other_chunk),  # never connected to anything
        ),
        edges=(
            GraphEdge(
                source_id=str(chunk.id),
                target_id=str(document_id),
                relationship_type=RelationshipType.BELONGS_TO,
            ),
        ),
    )

    with pytest.raises(OrphanNodeError):
        GraphValidator().validate(document_id, graph, [chunk, other_chunk], [])


def test_invalid_endpoint_type_raises_when_cites_targets_a_document() -> None:
    document_id = PaperId(uuid4())
    chunk = _chunk(document_id)
    graph = KnowledgeGraph(
        document_id=document_id,
        nodes=(_document_node(document_id), _knowledge_unit_node(chunk)),
        edges=(
            GraphEdge(
                source_id=str(chunk.id),
                target_id=str(document_id),  # CITES must target a KnowledgeUnit
                relationship_type=RelationshipType.CITES,
            ),
        ),
    )

    with pytest.raises(InvalidEdgeEndpointTypeError):
        GraphValidator().validate(document_id, graph, [chunk], [])


def test_completeness_raises_when_a_chunk_has_no_corresponding_node() -> None:
    document_id = PaperId(uuid4())
    chunk = _chunk(document_id)
    missing_chunk = _chunk(document_id, order=1)
    graph = KnowledgeGraph(
        document_id=document_id,
        nodes=(_document_node(document_id), _knowledge_unit_node(chunk)),
        edges=(
            GraphEdge(
                source_id=str(chunk.id),
                target_id=str(document_id),
                relationship_type=RelationshipType.BELONGS_TO,
            ),
        ),
    )

    with pytest.raises(GraphCompletenessError):
        GraphValidator().validate(document_id, graph, [chunk, missing_chunk], [])
