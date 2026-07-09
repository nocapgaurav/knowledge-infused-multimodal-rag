"""Builds GraphEdge objects from a document's knowledge units and relationships.

Deterministic and stateless: no semantic relationship is ever invented.
`BELONGS_TO` and `NEXT` are derived directly from `Chunk` fields Module 5
already set (`section_id`, `paper_id`, `order`); `CITES`/`REFERENCES`/
`CONTINUES` are 1:1 projections of `backend.domain.Relationship` records
Module 5 already computed -- pure format translation, no new inference.
"""

from collections.abc import Sequence

from backend.domain import Chunk, PaperId, Relationship
from backend.domain import RelationshipType as DomainRelationshipType
from backend.graph.models import GraphEdge, RelationshipType

_DOMAIN_TO_GRAPH_RELATIONSHIP: dict[DomainRelationshipType, RelationshipType] = {
    DomainRelationshipType.CITES: RelationshipType.CITES,
    DomainRelationshipType.REFERENCES: RelationshipType.REFERENCES,
    DomainRelationshipType.CONTINUES: RelationshipType.CONTINUES,
}


def build_containment_edges(document_id: PaperId, chunks: Sequence[Chunk]) -> list[GraphEdge]:
    """Build BELONGS_TO edges from every chunk (and every section) to its container.

    Every knowledge unit gets a direct edge to the document, regardless of
    whether it also has a section -- so "which document is this chunk
    from" is always one hop, never a variable-length traversal. The
    inverse (`CONTAINS`) is deliberately not also built: a property graph
    traverses either direction of one stored edge natively.

    Args:
        document_id: Identifier of the document.
        chunks: The document's knowledge units.

    Returns:
        BELONGS_TO edges: KnowledgeUnit -> Section (if any), KnowledgeUnit
        -> Document (always), and Section -> Document (once per distinct section).
    """
    edges: list[GraphEdge] = []
    seen_sections: set[str] = set()
    document_node_id = str(document_id)

    for chunk in chunks:
        chunk_id = str(chunk.id)
        edges.append(
            GraphEdge(
                source_id=chunk_id,
                target_id=document_node_id,
                relationship_type=RelationshipType.BELONGS_TO,
            )
        )
        if chunk.section_id is None:
            continue
        section_id = str(chunk.section_id)
        edges.append(
            GraphEdge(
                source_id=chunk_id,
                target_id=section_id,
                relationship_type=RelationshipType.BELONGS_TO,
            )
        )
        if section_id not in seen_sections:
            seen_sections.add(section_id)
            edges.append(
                GraphEdge(
                    source_id=section_id,
                    target_id=document_node_id,
                    relationship_type=RelationshipType.BELONGS_TO,
                )
            )
    return edges


def build_sequence_edges(chunks: Sequence[Chunk]) -> list[GraphEdge]:
    """Build NEXT edges chaining chunks in document reading order.

    Args:
        chunks: The document's knowledge units.

    Returns:
        One NEXT edge per consecutive pair, ordered by `Chunk.order`.
    """
    ordered = sorted(chunks, key=lambda chunk: chunk.order)
    return [
        GraphEdge(
            source_id=str(current.id),
            target_id=str(following.id),
            relationship_type=RelationshipType.NEXT,
        )
        for current, following in zip(ordered, ordered[1:], strict=False)
    ]


def build_relationship_edges(relationships: Sequence[Relationship]) -> list[GraphEdge]:
    """Project Module 5's already-computed relationships into graph edges.

    Args:
        relationships: The document's citation/reference/continuation
            relationships, as computed by Module 5.

    Returns:
        One `GraphEdge` per relationship, a 1:1 mapping with no new inference.
    """
    return [
        GraphEdge(
            source_id=str(relationship.source_chunk_id),
            target_id=str(relationship.target_chunk_id),
            relationship_type=_DOMAIN_TO_GRAPH_RELATIONSHIP[relationship.relationship_type],
        )
        for relationship in relationships
    ]
