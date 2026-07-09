"""Relationship: an explicit, typed connection between two chunks.

This is what distinguishes a knowledge representation from plain chunking:
connections between chunks (a citation, an in-text reference to a figure or
table, a link between artificially split siblings) are reified as their own
persisted facts, not left implicit in text for a later module to re-infer.
"""

from enum import StrEnum

from pydantic import Field

from backend.domain.base import DomainModel
from backend.domain.identifiers import ChunkId, PaperId, RelationshipId, generate_id


class RelationshipType(StrEnum):
    """The kind of connection a `Relationship` represents.

    Deliberately narrow. Each member is produced by a mechanical,
    high-precision detection method (exact matching against known entities
    in the same paper) -- not inferred semantics. Types like "explains" or
    "supports" are not included: they would require genuine language
    understanding to detect reliably, and a wrong semantic edge actively
    misleads a downstream knowledge graph, which is worse than no edge.

    Containment (chunk belongs to section) and sequence (chunk follows
    chunk) are also deliberately not relationship types -- both are already
    captured, more efficiently, by `Chunk.section_id` and `Chunk.order`.
    """

    CITES = "cites"
    """Source chunk's text contains an in-text citation marker resolved to
    the target chunk (a reference entry)."""

    REFERENCES = "references"
    """Source chunk's text mentions a figure or table by label (e.g.
    "Figure 1"), resolved to the target chunk (that figure or table)."""

    CONTINUES = "continues"
    """Source and target chunks are sequential siblings produced by
    splitting one oversized paragraph; target is the next part."""


class Relationship(DomainModel):
    """An explicit, directed connection between two chunks in the same paper.

    Scoped to a single paper: both chunks always belong to the same
    `paper_id`. Cross-document relationships (e.g. once `Reference
    .resolved_paper_id` is populated by a future resolution step) are a
    natural extension but are out of scope here, since this module
    processes one paper at a time.

    Attributes:
        id: Unique identifier for this relationship.
        paper_id: Identifier of the paper both chunks belong to.
        source_chunk_id: Identifier of the chunk the connection originates from.
        target_chunk_id: Identifier of the chunk the connection points to.
        relationship_type: The kind of connection this represents.
    """

    id: RelationshipId = Field(default_factory=lambda: RelationshipId(generate_id()))
    paper_id: PaperId
    source_chunk_id: ChunkId
    target_chunk_id: ChunkId
    relationship_type: RelationshipType
