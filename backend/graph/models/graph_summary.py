"""GraphSummary: a snapshot of what a KnowledgeGraphStore currently holds for a document."""

from pydantic import BaseModel, ConfigDict, Field


class GraphSummary(BaseModel):
    """Node and edge counts actually present in the store for one document.

    Distinct from `KnowledgeGraph.node_count`/`relationship_count`, which
    describe an in-memory, not-yet-persisted graph -- this is a live read
    from the store itself, used to verify a write actually took effect.

    Attributes:
        node_count: Number of nodes currently stored for this document.
        relationship_count: Number of edges currently stored for this document.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_count: int = Field(ge=0)
    relationship_count: int = Field(ge=0)
