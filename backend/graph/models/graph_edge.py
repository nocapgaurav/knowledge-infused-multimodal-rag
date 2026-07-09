"""GraphEdge: a vendor-independent, directed relationship in the knowledge graph."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from backend.graph.models.graph_node import GraphPropertyValue


class RelationshipType(StrEnum):
    """The closed vocabulary of edge types this module ever produces.

    Every member here is deterministic: either a direct projection of a
    structural fact already present on `Chunk` (`BELONGS_TO`, `NEXT`), or a
    1:1 projection of an already-computed `backend.domain.Relationship`
    (`CITES`, `REFERENCES`, `CONTINUES`). No semantic relationship type
    (`SUPPORTS`, `SIMILAR_TO`, ...) belongs in this enum -- see the Phase 1
    architectural review for why those are out of scope for this module.
    """

    BELONGS_TO = "BELONGS_TO"
    """Child points to its container: KnowledgeUnit -> Section,
    KnowledgeUnit -> Document, or Section -> Document. The inverse
    (`CONTAINS`) is deliberately not also materialized -- a property graph
    traverses either direction of one stored edge natively, so a second,
    reverse edge would double the edge count for no new fact."""

    NEXT = "NEXT"
    """KnowledgeUnit -> KnowledgeUnit, linking consecutive `Chunk.order`
    values across the whole document. Gives O(1) reading-order neighbor
    expansion without sorting by `order`."""

    CITES = "CITES"
    """Direct projection of `backend.domain.RelationshipType.CITES`."""

    REFERENCES = "REFERENCES"
    """Direct projection of `backend.domain.RelationshipType.REFERENCES`."""

    CONTINUES = "CONTINUES"
    """Direct projection of `backend.domain.RelationshipType.CONTINUES`."""


class GraphEdge(BaseModel):
    """A single directed edge in the knowledge graph, independent of any database.

    Attributes:
        source_id: Id of the node this edge originates from.
        target_id: Id of the node this edge points to.
        relationship_type: The kind of connection this edge represents.
        properties: Flat property bag. Empty for every edge type this
            module currently produces; kept for forward compatibility so a
            future edge type can carry data without a model change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    relationship_type: RelationshipType
    properties: dict[str, GraphPropertyValue] = Field(default_factory=dict)

    @property
    def identity_key(self) -> tuple[str, str, RelationshipType]:
        """Return the `(source_id, target_id, relationship_type)` triple.

        Two edges sharing this triple are duplicates -- used by
        `GraphValidator` for duplicate-edge detection.
        """
        return (self.source_id, self.target_id, self.relationship_type)
