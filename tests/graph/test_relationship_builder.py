"""Tests for deriving GraphEdge objects from a document's knowledge units and relationships."""

from uuid import uuid4

from backend.domain import Chunk, ChunkModality, PaperId, Relationship, SectionId
from backend.domain import RelationshipType as DomainRelationshipType
from backend.graph.builders.relationship_builder import (
    build_containment_edges,
    build_relationship_edges,
    build_sequence_edges,
)
from backend.graph.models import RelationshipType


def _chunk(
    paper_id: PaperId,
    order: int,
    section_id: SectionId | None = None,
    modality: ChunkModality = ChunkModality.TEXT,
) -> Chunk:
    return Chunk(
        paper_id=paper_id, section_id=section_id, order=order, modality=modality, text="text"
    )


def test_every_chunk_gets_a_belongs_to_edge_to_the_document() -> None:
    document_id = PaperId(uuid4())
    chunks = [_chunk(document_id, 0), _chunk(document_id, 1)]

    edges = build_containment_edges(document_id, chunks)

    document_edges = [edge for edge in edges if edge.target_id == str(document_id)]
    assert {edge.source_id for edge in document_edges} == {str(chunk.id) for chunk in chunks}
    assert all(edge.relationship_type is RelationshipType.BELONGS_TO for edge in document_edges)


def test_chunk_with_section_gets_belongs_to_section_and_section_gets_belongs_to_document() -> None:
    document_id = PaperId(uuid4())
    section_id = SectionId(uuid4())
    chunk = _chunk(document_id, 0, section_id=section_id)

    edges = build_containment_edges(document_id, [chunk])

    chunk_to_section = [
        e for e in edges if e.source_id == str(chunk.id) and e.target_id == str(section_id)
    ]
    section_to_document = [
        e for e in edges if e.source_id == str(section_id) and e.target_id == str(document_id)
    ]
    assert len(chunk_to_section) == 1
    assert len(section_to_document) == 1


def test_section_to_document_edge_is_not_duplicated_across_chunks_in_the_same_section() -> None:
    document_id = PaperId(uuid4())
    section_id = SectionId(uuid4())
    chunks = [
        _chunk(document_id, 0, section_id=section_id),
        _chunk(document_id, 1, section_id=section_id),
    ]

    edges = build_containment_edges(document_id, chunks)

    section_to_document = [
        e for e in edges if e.source_id == str(section_id) and e.target_id == str(document_id)
    ]
    assert len(section_to_document) == 1


def test_no_contains_or_has_figure_edges_are_built() -> None:
    document_id = PaperId(uuid4())
    chunk = _chunk(document_id, 0, modality=ChunkModality.FIGURE)

    edges = build_containment_edges(document_id, [chunk])

    assert all(edge.relationship_type is RelationshipType.BELONGS_TO for edge in edges)


def test_sequence_edges_chain_chunks_in_order_regardless_of_input_order() -> None:
    document_id = PaperId(uuid4())
    chunk0 = _chunk(document_id, 0)
    chunk1 = _chunk(document_id, 1)
    chunk2 = _chunk(document_id, 2)

    edges = build_sequence_edges([chunk2, chunk0, chunk1])  # shuffled input

    assert [(e.source_id, e.target_id) for e in edges] == [
        (str(chunk0.id), str(chunk1.id)),
        (str(chunk1.id), str(chunk2.id)),
    ]
    assert all(edge.relationship_type is RelationshipType.NEXT for edge in edges)


def test_sequence_edges_empty_for_single_chunk() -> None:
    document_id = PaperId(uuid4())
    assert build_sequence_edges([_chunk(document_id, 0)]) == []


def test_relationship_edges_are_a_1to1_projection() -> None:
    document_id = PaperId(uuid4())
    source_id = uuid4()
    target_id = uuid4()
    relationship = Relationship(
        paper_id=document_id,
        source_chunk_id=source_id,
        target_chunk_id=target_id,
        relationship_type=DomainRelationshipType.CITES,
    )

    edges = build_relationship_edges([relationship])

    assert len(edges) == 1
    assert edges[0].source_id == str(source_id)
    assert edges[0].target_id == str(target_id)
    assert edges[0].relationship_type is RelationshipType.CITES


def test_all_three_domain_relationship_types_map_correctly() -> None:
    document_id = PaperId(uuid4())
    relationships = [
        Relationship(
            paper_id=document_id,
            source_chunk_id=uuid4(),
            target_chunk_id=uuid4(),
            relationship_type=domain_type,
        )
        for domain_type in DomainRelationshipType
    ]

    edges = build_relationship_edges(relationships)

    assert {edge.relationship_type for edge in edges} == {
        RelationshipType.CITES,
        RelationshipType.REFERENCES,
        RelationshipType.CONTINUES,
    }
