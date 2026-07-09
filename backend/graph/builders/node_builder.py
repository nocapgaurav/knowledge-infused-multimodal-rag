"""Builds GraphNode objects from a document's knowledge units.

Deterministic and stateless: the same chunks always produce the same
nodes, since every node id is the source entity's own id, never freshly
generated.
"""

from collections.abc import Sequence

from backend.domain import Chunk, ChunkModality, PaperId
from backend.graph.models import GraphNode, NodeLabel

_MODALITY_LABELS: dict[ChunkModality, NodeLabel] = {
    ChunkModality.TEXT: NodeLabel.TEXT_UNIT,
    ChunkModality.TABLE: NodeLabel.TABLE_UNIT,
    ChunkModality.FIGURE: NodeLabel.FIGURE_UNIT,
}


def build_document_node(document_id: PaperId) -> GraphNode:
    """Build the single root node representing the document itself.

    Args:
        document_id: Identifier of the document.

    Returns:
        A `GraphNode` labeled `DOCUMENT`. Carries `paper_id` (equal to its
        own `id`) alongside every other node type so a store's "delete this
        document's whole graph" query can filter on one property uniformly,
        instead of needing a label-specific case for the root node.
    """
    return GraphNode(
        id=str(document_id),
        labels=(NodeLabel.DOCUMENT,),
        properties={"id": str(document_id), "paper_id": str(document_id)},
    )


def build_section_nodes(document_id: PaperId, chunks: Sequence[Chunk]) -> list[GraphNode]:
    """Build one node per distinct section referenced by the document's chunks.

    Deliberately identity-only: no title or level, since neither is present
    on `Chunk` -- see the Phase 1 architectural review for why this module
    never reads `paper.json` to obtain them. The node exists purely so
    `BELONGS_TO` gives real graph traversal, and so a future module can
    attach richer section metadata later without a schema migration.

    Args:
        document_id: Identifier of the document.
        chunks: The document's knowledge units.

    Returns:
        One `GraphNode` per distinct non-null `section_id`, in first-seen order.
    """
    seen: dict[str, GraphNode] = {}
    for chunk in chunks:
        if chunk.section_id is None:
            continue
        section_id = str(chunk.section_id)
        if section_id not in seen:
            seen[section_id] = GraphNode(
                id=section_id,
                labels=(NodeLabel.SECTION,),
                properties={"id": section_id, "paper_id": str(document_id)},
            )
    return list(seen.values())


def build_knowledge_unit_nodes(chunks: Sequence[Chunk]) -> list[GraphNode]:
    """Build one node per knowledge unit, labeled by its modality.

    No separate `Figure`/`Table`/`Reference` node types are built: Module 5
    already fused those into `Chunk`, so a modality-specific secondary
    label (`TextUnit`/`TableUnit`/`FigureUnit`) alongside the shared
    `KnowledgeUnit` label gives both uniform and type-filtered queries
    without re-fragmenting what Module 5 deliberately unified.

    Args:
        chunks: The document's knowledge units.

    Returns:
        One `GraphNode` per chunk, carrying `KNOWLEDGE_UNIT` plus its
        modality-specific secondary label.
    """
    return [
        GraphNode(
            id=str(chunk.id),
            labels=(NodeLabel.KNOWLEDGE_UNIT, _MODALITY_LABELS[chunk.modality]),
            properties={
                "id": str(chunk.id),
                "paper_id": str(chunk.paper_id),
                "section_id": str(chunk.section_id) if chunk.section_id is not None else None,
                "order": chunk.order,
                "modality": chunk.modality.value,
            },
        )
        for chunk in chunks
    ]
