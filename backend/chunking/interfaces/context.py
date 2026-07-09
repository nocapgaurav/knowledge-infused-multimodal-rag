"""Shared data contracts strategies and the builder communicate through.

Deliberately not a behavioral interface (no `ChunkingStrategy` ABC): the
paragraph strategy operates on a list of paragraphs (it needs lookahead to
merge tiny fragments), while the figure, table, and reference strategies
each operate on a single entity. Forcing these into one generic interface
would be a hollow abstraction with no real polymorphic caller anywhere in
this design.
"""

from dataclasses import dataclass, field
from uuid import UUID

from backend.domain import Chunk, ChunkId, FigureId, Paper, ReferenceId, Relationship, TableId


@dataclass
class _OrderCounter:
    """A single incrementing counter, shared by every strategy invocation.

    `Chunk.order` must be globally sequential across the whole paper's
    knowledge units, regardless of which strategy or content type produced
    them -- so the counter is owned by one place (`BuildContext`), not
    reimplemented per strategy.
    """

    _next: int = 0

    def next(self) -> int:
        value = self._next
        self._next += 1
        return value


@dataclass
class BuildContext:
    """Shared, paper-level context available to every strategy.

    Constructed once per paper and threaded through both build passes
    (chunk construction, then relationship detection) -- lookups derived
    directly from `paper` are computed once up front; `entity_chunk_ids` is
    populated incrementally as chunks are created and read back during
    relationship detection.

    Attributes:
        paper: The paper being represented.
        max_words_per_chunk: Word-count threshold above which a paragraph
            is split at sentence boundaries.
        min_words_per_chunk: Word-count floor below which a paragraph is
            merged into a neighbor rather than becoming its own unit.
        figure_number_lookup: Maps a figure's printed number (parsed from
            its caption, e.g. "Figure 1" -> 1) to its domain id.
        table_number_lookup: Maps a table's printed number (parsed from its
            caption, e.g. "Table 2" -> 2) to its domain id.
        reference_number_lookup: Maps a reference's citation number
            (its bibliography position, 1-indexed) to its domain id.
        entity_chunk_ids: Maps a source entity's id (figure, table,
            reference, or paragraph) to the id of the chunk built from it.
            Populated during chunk construction; read during relationship
            detection, once every chunk exists.
    """

    paper: Paper
    max_words_per_chunk: int
    min_words_per_chunk: int
    figure_number_lookup: dict[int, FigureId]
    table_number_lookup: dict[int, TableId]
    reference_number_lookup: dict[int, ReferenceId]
    entity_chunk_ids: dict[UUID, ChunkId] = field(default_factory=dict)
    _order_counter: _OrderCounter = field(default_factory=_OrderCounter)

    def next_order(self) -> int:
        """Return the next sequential order value for a new chunk."""
        return self._order_counter.next()

    def register_chunk(self, entity_id: UUID, chunk_id: ChunkId) -> None:
        """Record which chunk was built from a given source entity.

        Args:
            entity_id: Identifier of the source entity (figure, table,
                reference, or paragraph) a chunk was built from.
            chunk_id: Identifier of the resulting chunk.
        """
        self.entity_chunk_ids[entity_id] = chunk_id


@dataclass(frozen=True)
class StrategyResult:
    """The chunks and relationships produced by one strategy invocation.

    Attributes:
        chunks: Chunks produced.
        relationships: Relationships produced (e.g. `CONTINUES` links
            between split paragraph siblings). Empty for strategies that
            don't detect relationships during construction.
    """

    chunks: list[Chunk]
    relationships: list[Relationship]
