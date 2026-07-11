"""Which relationship types graph expansion follows, in which directions,
and how much each is trusted.

This is the concrete mechanism behind "graph expansion should improve
evidence quality, not merely increase recall" (Phase 1 architectural
review, Q7): each relationship type is deterministic and already computed
(by Module 5 or Module 8), but they carry different precision. `CITES`/
`REFERENCES` are explicit, high-precision citations detected by exact
matching against known entities -- when they lead somewhere, that
destination is almost certainly relevant. `BELONGS_TO` connects everything
in a section indiscriminately -- it is a useful discovery mechanism but a
weak relevance signal on its own. Confidence tiers encode this directly,
and are reused by `evaluation/evidence_evaluator.py`'s relationship
confidence signal.
"""

from backend.retrieval.models import TraversalDirection

RELATIONSHIP_TYPES: tuple[str, ...] = ("CITES", "REFERENCES", "CONTINUES", "NEXT", "BELONGS_TO")
"""Every relationship type expansion follows, ordered highest-confidence
first -- this order is also the truncation priority when a node's neighbor
count exceeds `ExpansionBudget.max_neighbors_per_node` (see
`graph_expander.py`), so a budget cut preferentially keeps high-precision
edges over broad ones."""

_DIRECTIONS: dict[str, tuple[TraversalDirection, ...]] = {
    "CITES": (TraversalDirection.OUTGOING,),
    # Bidirectional: outgoing finds the figure/table a seed paragraph
    # mentions; INCOMING finds every paragraph that mentions a seed
    # figure/table. Observed live in Sprint 2: "What is Figure 2?" seeded
    # the figure chunk but retrieved none of the paragraphs discussing it,
    # leaving the model nothing to explain beyond the caption.
    "REFERENCES": (TraversalDirection.OUTGOING, TraversalDirection.INCOMING),
    "CONTINUES": (TraversalDirection.OUTGOING, TraversalDirection.INCOMING),
    "NEXT": (TraversalDirection.OUTGOING, TraversalDirection.INCOMING),
    "BELONGS_TO": (TraversalDirection.OUTGOING, TraversalDirection.INCOMING),
}

_CONFIDENCE_TIER: dict[str, int] = {
    "CITES": 3,
    "REFERENCES": 3,
    "CONTINUES": 2,
    "NEXT": 2,
    "BELONGS_TO": 1,
}


def directions_for(relationship_type: str) -> tuple[TraversalDirection, ...]:
    """Return which directions of a relationship type expansion follows.

    Args:
        relationship_type: Name of the relationship type.

    Returns:
        The directions to follow, or an empty tuple for an unrecognized type.
    """
    return _DIRECTIONS.get(relationship_type, ())


def confidence_tier(relationship_type: str) -> int:
    """Return how much a relationship type's edges should be trusted.

    Higher is more trustworthy. A direct dense match (no relationship at
    all) is intentionally not modeled here -- it is handled as its own,
    highest tier by the evaluator, since it needed no graph corroboration.

    Args:
        relationship_type: Name of the relationship type.

    Returns:
        The confidence tier, or `0` for an unrecognized type.
    """
    return _CONFIDENCE_TIER.get(relationship_type, 0)
