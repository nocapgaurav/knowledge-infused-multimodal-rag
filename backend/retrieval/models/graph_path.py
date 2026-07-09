"""GraphPath: the deterministic trace of how a candidate was reached via traversal."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TraversalDirection(StrEnum):
    """Which direction of a relationship a traversal hop followed.

    A property graph edge is stored once, directed; a traversal can follow
    it forward (`OUTGOING`), backward (`INCOMING`), or a caller can ask for
    both without caring which. Recorded per hop so a path can be replayed
    exactly, not just its endpoints.
    """

    OUTGOING = "outgoing"
    INCOMING = "incoming"


class TraversalHop(BaseModel):
    """One single-edge step in a graph traversal.

    Attributes:
        source_id: Id of the node this hop started from.
        target_id: Id of the node this hop arrived at.
        relationship_type: Name of the graph relationship followed (e.g.
            "NEXT", "CITES") -- a plain string, not Module 8's
            `RelationshipType`, since this module must not import Module
            8's internal vocabulary; it only ever sees what
            `GraphRetriever` reports.
        direction: Which direction of the relationship was followed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    relationship_type: str = Field(min_length=1)
    direction: TraversalDirection


class GraphNeighbor(BaseModel):
    """One single-hop neighbor reported by a `GraphRetriever`.

    The raw provider primitive `GraphExpander` consumes to build up
    validated `TraversalHop`s -- distinct from `TraversalHop` because a
    provider reports every neighbor it finds, including ones the expander
    will discard (already visited, budget exhausted, wrong node type).

    Attributes:
        source_id: Id of the node this neighbor was found from.
        neighbor_id: Id of the neighboring node.
        neighbor_labels: The neighboring node's labels (e.g. `("KnowledgeUnit",
            "FigureUnit")`), so the caller can decide whether to treat it as
            evidence or as an intermediate node to keep traversing through.
        relationship_type: Name of the relationship connecting them.
        direction: Which direction of the relationship was followed to find it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1)
    neighbor_id: str = Field(min_length=1)
    neighbor_labels: tuple[str, ...] = Field(min_length=1)
    relationship_type: str = Field(min_length=1)
    direction: TraversalDirection


class GraphPath(BaseModel):
    """The complete, ordered sequence of hops from a seed candidate to a
    graph-discovered candidate.

    Empty for a candidate discovered directly by dense retrieval -- it was
    never reached by traversal at all.

    Attributes:
        hops: Ordered traversal steps, first hop first. `hops[-1].target_id`
            is the discovered candidate's own node id.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hops: tuple[TraversalHop, ...] = Field(default_factory=tuple)

    @property
    def depth(self) -> int:
        """Number of hops in this path -- `0` for a dense-only candidate."""
        return len(self.hops)
