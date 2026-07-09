"""Tests for deriving GraphNode objects from a document's knowledge units."""

from uuid import uuid4

from backend.domain import Chunk, ChunkModality, PaperId, SectionId
from backend.graph.builders.node_builder import (
    build_document_node,
    build_knowledge_unit_nodes,
    build_section_nodes,
)
from backend.graph.models import NodeLabel


def _chunk(
    paper_id: PaperId,
    order: int,
    modality: ChunkModality = ChunkModality.TEXT,
    section_id: SectionId | None = None,
) -> Chunk:
    return Chunk(
        paper_id=paper_id,
        section_id=section_id,
        order=order,
        modality=modality,
        text="some text",
    )


def test_build_document_node_carries_id_and_paper_id_as_properties() -> None:
    document_id = PaperId(uuid4())

    node = build_document_node(document_id)

    assert node.id == str(document_id)
    assert node.labels == (NodeLabel.DOCUMENT,)
    assert node.properties == {"id": str(document_id), "paper_id": str(document_id)}


def test_build_section_nodes_deduplicates_by_section_id() -> None:
    document_id = PaperId(uuid4())
    section_id = SectionId(uuid4())
    chunks = [
        _chunk(document_id, 0, section_id=section_id),
        _chunk(document_id, 1, section_id=section_id),
        _chunk(document_id, 2, section_id=None),
    ]

    nodes = build_section_nodes(document_id, chunks)

    assert len(nodes) == 1
    assert nodes[0].id == str(section_id)
    assert nodes[0].labels == (NodeLabel.SECTION,)
    assert nodes[0].properties == {"id": str(section_id), "paper_id": str(document_id)}


def test_build_section_nodes_returns_empty_for_no_sections() -> None:
    document_id = PaperId(uuid4())
    chunks = [_chunk(document_id, 0)]

    assert build_section_nodes(document_id, chunks) == []


def test_knowledge_unit_node_carries_knowledge_unit_and_modality_labels() -> None:
    document_id = PaperId(uuid4())
    section_id = SectionId(uuid4())
    chunk = _chunk(document_id, 0, modality=ChunkModality.FIGURE, section_id=section_id)

    nodes = build_knowledge_unit_nodes([chunk])

    assert len(nodes) == 1
    node = nodes[0]
    assert node.id == str(chunk.id)
    assert node.labels == (NodeLabel.KNOWLEDGE_UNIT, NodeLabel.FIGURE_UNIT)
    assert node.properties["section_id"] == str(section_id)
    assert node.properties["order"] == 0
    assert node.properties["modality"] == "figure"


def test_knowledge_unit_node_section_id_is_none_when_chunk_has_no_section() -> None:
    document_id = PaperId(uuid4())
    chunk = _chunk(document_id, 0, section_id=None)

    nodes = build_knowledge_unit_nodes([chunk])

    assert nodes[0].properties["section_id"] is None


def test_each_modality_maps_to_the_correct_secondary_label() -> None:
    document_id = PaperId(uuid4())
    chunks = [
        _chunk(document_id, 0, modality=ChunkModality.TEXT),
        _chunk(document_id, 1, modality=ChunkModality.TABLE),
        _chunk(document_id, 2, modality=ChunkModality.FIGURE),
    ]

    nodes = build_knowledge_unit_nodes(chunks)

    assert [node.labels[1] for node in nodes] == [
        NodeLabel.TEXT_UNIT,
        NodeLabel.TABLE_UNIT,
        NodeLabel.FIGURE_UNIT,
    ]


def test_construction_is_deterministic_across_repeated_calls() -> None:
    document_id = PaperId(uuid4())
    chunk = _chunk(document_id, 0)

    first = build_knowledge_unit_nodes([chunk])
    second = build_knowledge_unit_nodes([chunk])

    assert first == second
