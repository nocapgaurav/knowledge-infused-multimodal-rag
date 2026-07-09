"""Orchestrates knowledge representation construction.

Two passes:

1. A single forward walk over the paper's reading order (abstract first,
   then sections in reading order, then references last), dispatching each
   item to its strategy and constructing every chunk. `CONTINUES`
   relationships (split-paragraph siblings) are produced here too, since
   they only need local knowledge.
2. A second pass over paragraph-derived chunks only, detecting citations
   and figure/table mentions -- deferred because a citation can point at a
   reference that (in reading order) appears after the citing paragraph,
   so every chunk must exist before resolution is attempted.
"""

import logging
import re
from dataclasses import dataclass

from backend.chunking.builder.reading_order import compute_reading_order
from backend.chunking.interfaces.context import BuildContext
from backend.chunking.strategies.caption_lookup import find_caption
from backend.chunking.strategies.figure_strategy import FigureStrategy
from backend.chunking.strategies.paragraph_strategy import ParagraphStrategy
from backend.chunking.strategies.simple_strategies import AbstractStrategy, ReferenceStrategy
from backend.chunking.strategies.table_strategy import TableStrategy
from backend.domain import (
    CaptionSubjectType,
    Chunk,
    FigureId,
    Paper,
    Paragraph,
    Relationship,
    Table,
    TableId,
)

logger = logging.getLogger(__name__)

_LEADING_NUMBER = re.compile(r"^\s*(?:Figure|Fig\.?|Table)\s+(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class BuildResult:
    """The complete knowledge representation produced for one paper.

    Attributes:
        chunks: Every knowledge unit produced, across all content types.
        relationships: Every relationship detected between them.
    """

    chunks: list[Chunk]
    relationships: list[Relationship]


class KnowledgeBuilder:
    """Builds the full knowledge representation (chunks + relationships) for a paper."""

    def __init__(
        self,
        max_words_per_chunk: int,
        min_words_per_chunk: int,
        paragraph_strategy: ParagraphStrategy | None = None,
        figure_strategy: FigureStrategy | None = None,
        table_strategy: TableStrategy | None = None,
        abstract_strategy: AbstractStrategy | None = None,
        reference_strategy: ReferenceStrategy | None = None,
    ) -> None:
        """Initialize the builder.

        Args:
            max_words_per_chunk: Word-count threshold above which a
                paragraph is split at sentence boundaries.
            min_words_per_chunk: Word-count floor below which a paragraph
                is merged into a neighbor rather than becoming its own unit.
            paragraph_strategy: Strategy for paragraph runs. Defaults to a
                new `ParagraphStrategy`.
            figure_strategy: Strategy for figures. Defaults to a new `FigureStrategy`.
            table_strategy: Strategy for tables. Defaults to a new `TableStrategy`.
            abstract_strategy: Strategy for the abstract. Defaults to a new `AbstractStrategy`.
            reference_strategy: Strategy for references. Defaults to a new `ReferenceStrategy`.
        """
        self._max_words_per_chunk = max_words_per_chunk
        self._min_words_per_chunk = min_words_per_chunk
        self._paragraph_strategy = paragraph_strategy or ParagraphStrategy()
        self._figure_strategy = figure_strategy or FigureStrategy()
        self._table_strategy = table_strategy or TableStrategy()
        self._abstract_strategy = abstract_strategy or AbstractStrategy()
        self._reference_strategy = reference_strategy or ReferenceStrategy()

    def build(self, paper: Paper) -> BuildResult:
        """Build the complete knowledge representation for a paper.

        Args:
            paper: The paper to represent.

        Returns:
            Every chunk and relationship produced.
        """
        context = self._make_context(paper)
        chunks: list[Chunk] = []
        relationships: list[Relationship] = []
        paragraph_chunks: list[Chunk] = []
        paragraph_buffer: list[Paragraph] = []

        def flush_paragraphs() -> None:
            if not paragraph_buffer:
                return
            result = self._paragraph_strategy.build(list(paragraph_buffer), context)
            chunks.extend(result.chunks)
            relationships.extend(result.relationships)
            paragraph_chunks.extend(result.chunks)
            paragraph_buffer.clear()

        abstract_result = self._abstract_strategy.build(context)
        chunks.extend(abstract_result.chunks)
        relationships.extend(abstract_result.relationships)

        for ordered in compute_reading_order(paper):
            item = ordered.item
            if isinstance(item, Paragraph):
                if paragraph_buffer and paragraph_buffer[-1].section_id != item.section_id:
                    flush_paragraphs()
                paragraph_buffer.append(item)
                continue
            flush_paragraphs()
            if isinstance(item, Table):
                result = self._table_strategy.build(item, context)
            else:
                result = self._figure_strategy.build(item, context)
            chunks.extend(result.chunks)
            relationships.extend(result.relationships)
        flush_paragraphs()

        for reference in sorted(paper.references, key=lambda r: r.order):
            result = self._reference_strategy.build(reference, context)
            chunks.extend(result.chunks)
            relationships.extend(result.relationships)

        for paragraph_chunk in paragraph_chunks:
            relationships.extend(
                self._paragraph_strategy.detect_relationships(paragraph_chunk, context)
            )

        logger.info(
            "knowledge representation built",
            extra={
                "document_id": str(paper.id),
                "chunks": len(chunks),
                "relationships": len(relationships),
            },
        )
        return BuildResult(chunks=chunks, relationships=relationships)

    def _make_context(self, paper: Paper) -> BuildContext:
        return BuildContext(
            paper=paper,
            max_words_per_chunk=self._max_words_per_chunk,
            min_words_per_chunk=self._min_words_per_chunk,
            figure_number_lookup=_build_figure_number_lookup(paper),
            table_number_lookup=_build_table_number_lookup(paper),
            reference_number_lookup={
                reference.order + 1: reference.id for reference in paper.references
            },
        )


def _build_figure_number_lookup(paper: Paper) -> dict[int, FigureId]:
    lookup: dict[int, FigureId] = {}
    for figure in paper.figures:
        caption = find_caption(paper, figure.id, CaptionSubjectType.FIGURE)
        match = _LEADING_NUMBER.match(caption.text) if caption is not None else None
        if match:
            lookup.setdefault(int(match.group(1)), figure.id)
    return lookup


def _build_table_number_lookup(paper: Paper) -> dict[int, TableId]:
    lookup: dict[int, TableId] = {}
    for table in paper.tables:
        caption = find_caption(paper, table.id, CaptionSubjectType.TABLE)
        match = _LEADING_NUMBER.match(caption.text) if caption is not None else None
        if match:
            lookup.setdefault(int(match.group(1)), table.id)
    return lookup
