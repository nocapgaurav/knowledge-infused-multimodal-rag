"""GraphNode: a vendor-independent node in the knowledge graph."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

GraphPropertyValue = str | int | float | bool | None
"""A property value Neo4j (and any other property graph store) can store
directly -- scalars only. Nested objects have no equivalent in a property
graph's data model, so nothing here ever produces one."""


class NodeLabel(StrEnum):
    """The closed vocabulary of node labels this module ever produces.

    Closed rather than an arbitrary string, so a builder cannot silently
    introduce a new node type the validator and provider don't know about.
    """

    DOCUMENT = "Document"
    """One per paper -- the graph's root."""

    SECTION = "Section"
    """One per distinct `Chunk.section_id`. Deliberately identity-only (see
    `KnowledgeGraph` module docstring): no title or level, since those live
    in `paper.json`, which this module never reads."""

    KNOWLEDGE_UNIT = "KnowledgeUnit"
    """One per `Chunk`, regardless of modality. Always paired with exactly
    one modality-specific secondary label below -- never used alone."""

    TEXT_UNIT = "TextUnit"
    """Secondary label for a knowledge unit with `ChunkModality.TEXT`."""

    TABLE_UNIT = "TableUnit"
    """Secondary label for a knowledge unit with `ChunkModality.TABLE`."""

    FIGURE_UNIT = "FigureUnit"
    """Secondary label for a knowledge unit with `ChunkModality.FIGURE`."""


class GraphNode(BaseModel):
    """A single node in the knowledge graph, independent of any database.

    Attributes:
        id: Identifier of this node. Always the source domain entity's own
            id (a `Chunk.id`, `SectionId`, or `PaperId`) rendered as a
            string -- never freshly generated -- so rebuilding the graph
            from the same knowledge representation reproduces the same
            node ids every time.
        labels: This node's labels, most specific last. Always at least
            one label. A knowledge unit carries exactly two: `KNOWLEDGE_UNIT`
            followed by its modality-specific label.
        properties: Flat property bag. Deliberately minimal -- structural
            identity only (id, paper_id, section_id, order, modality), not
            chunk text or asset references, which already live in
            `knowledge_units.json` and the vector store; duplicating them
            here would be a second copy that can silently drift out of sync.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    labels: tuple[NodeLabel, ...] = Field(min_length=1)
    properties: dict[str, GraphPropertyValue] = Field(default_factory=dict)

    @field_validator("labels")
    @classmethod
    def _labels_have_no_duplicates(cls, labels: tuple[NodeLabel, ...]) -> tuple[NodeLabel, ...]:
        if len(set(labels)) != len(labels):
            raise ValueError(f"labels must not contain duplicates: {labels}")
        return labels
