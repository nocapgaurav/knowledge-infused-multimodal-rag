"""Turns a run of same-section paragraphs into knowledge units.

Two passes live here, matching the builder's own two-pass design:

- `build` (pass 1): merges degenerate tiny fragments into a neighbor,
  splits oversized paragraphs at sentence boundaries, and constructs
  chunks. Runs during the builder's single forward walk.
- `detect_relationships` (pass 2): finds in-text citations and figure/table
  mentions. Runs after every chunk in the paper exists, since a citation
  can point at a reference that (in reading order) appears after the
  citing paragraph.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from backend.chunking.interfaces.context import BuildContext, StrategyResult
from backend.chunking.strategies.numbering import (
    NUMBER_TOKEN_PATTERN,
    parse_printed_number,
)
from backend.domain import (
    BoundingBox,
    Chunk,
    ChunkId,
    ChunkModality,
    Paragraph,
    Relationship,
    RelationshipType,
    SectionId,
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_FIGURE_MENTION_PATTERN = re.compile(
    r"\b(?:Figure|Fig\.?)\s+" + NUMBER_TOKEN_PATTERN, re.IGNORECASE
)
_TABLE_MENTION_PATTERN = re.compile(r"\bTable\s+" + NUMBER_TOKEN_PATTERN, re.IGNORECASE)


def _section_context(section_id: SectionId | None, context: BuildContext) -> str | None:
    """The section-title identity a body chunk carries into retrieval.

    Embedding a paragraph together with the title of the section it lives
    in lets section-level questions ("summarize the methodology") land on
    the right passages, and gives every evidence card a human-readable
    origin instead of a bare knowledge-unit id. `None` when the paragraph
    has no section (front matter is labeled separately by the builder).
    """
    if section_id is None:
        return None
    section = next((s for s in context.paper.sections if s.id == section_id), None)
    if section is None:
        return None
    return f"Section: {section.title}"


def _word_count(text: str) -> int:
    return len(text.split())


@dataclass
class _MergedBlock:
    """One or more original paragraphs, combined by the tiny-fragment merge pass."""

    source_paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(p.text for p in self.source_paragraphs)

    @property
    def bounding_boxes(self) -> list[BoundingBox]:
        return [box for p in self.source_paragraphs for box in p.bounding_boxes]

    def merge_with(self, other: "_MergedBlock") -> "_MergedBlock":
        return _MergedBlock(source_paragraphs=[*self.source_paragraphs, *other.source_paragraphs])


def _merge_tiny_paragraphs(paragraphs: Sequence[Paragraph], min_words: int) -> list[_MergedBlock]:
    """Merge paragraphs below `min_words` into a neighbor rather than leaving
    them as their own low-information retrieval unit.

    A tiny paragraph merges forward into the next one; a tiny paragraph
    with nothing after it (the last in the section) merges backward into
    the previous result instead. A section consisting entirely of tiny
    paragraphs is kept as a single merged block.
    """
    result: list[_MergedBlock] = []
    pending: _MergedBlock | None = None

    for paragraph in paragraphs:
        block = _MergedBlock([paragraph])
        if pending is not None:
            block = pending.merge_with(block)
            pending = None
        if _word_count(block.text) < min_words:
            pending = block
        else:
            result.append(block)

    if pending is not None:
        if result:
            result[-1] = result[-1].merge_with(pending)
        else:
            result.append(pending)

    return result


def _split_oversized_block(text: str, max_words: int) -> list[str]:
    """Split text at sentence boundaries so no part exceeds `max_words`.

    Never cuts mid-sentence, even if a single sentence alone exceeds
    `max_words` -- that sentence simply becomes its own oversized part.
    Adjacent parts share a one-sentence overlap, since this is the one
    place the split boundary is artificial (introduced by this module, not
    the author).
    """
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    if not sentences:
        return [text] if text.strip() else []

    parts: list[str] = []
    current = [sentences[0]]
    current_words = _word_count(sentences[0])

    for sentence in sentences[1:]:
        sentence_words = _word_count(sentence)
        if current_words + sentence_words > max_words:
            parts.append(" ".join(current))
            overlap_sentence = current[-1]
            current = [overlap_sentence, sentence]
            current_words = _word_count(overlap_sentence) + sentence_words
        else:
            current.append(sentence)
            current_words += sentence_words

    parts.append(" ".join(current))
    return parts


class ParagraphStrategy:
    """Builds knowledge units from a run of same-section paragraphs."""

    def build(self, paragraphs: Sequence[Paragraph], context: BuildContext) -> StrategyResult:
        """Build chunks for a contiguous run of paragraphs from the same section.

        Args:
            paragraphs: Paragraphs to process, in reading order, all
                sharing the same `section_id`.
            context: Shared build context.

        Returns:
            The chunks built, plus any `CONTINUES` relationships between
            artificially split siblings.
        """
        if not paragraphs:
            return StrategyResult(chunks=[], relationships=[])

        paper_id = context.paper.id
        section_id = paragraphs[0].section_id
        retrieval_context = _section_context(section_id, context)
        chunks: list[Chunk] = []
        relationships: list[Relationship] = []

        for block in _merge_tiny_paragraphs(paragraphs, context.min_words_per_chunk):
            part_chunk_ids: list[ChunkId] = []
            for part_text in _split_oversized_block(block.text, context.max_words_per_chunk):
                chunk = Chunk(
                    paper_id=paper_id,
                    section_id=section_id,
                    order=context.next_order(),
                    modality=ChunkModality.TEXT,
                    text=part_text,
                    retrieval_context=retrieval_context,
                    token_count=_word_count(part_text),
                    source_element_ids=[p.id for p in block.source_paragraphs],
                    bounding_boxes=block.bounding_boxes,
                )
                chunks.append(chunk)
                part_chunk_ids.append(chunk.id)

            for source_id, target_id in zip(part_chunk_ids, part_chunk_ids[1:], strict=False):
                relationships.append(
                    Relationship(
                        paper_id=paper_id,
                        source_chunk_id=source_id,
                        target_chunk_id=target_id,
                        relationship_type=RelationshipType.CONTINUES,
                    )
                )

        return StrategyResult(chunks=chunks, relationships=relationships)

    def detect_relationships(self, chunk: Chunk, context: BuildContext) -> list[Relationship]:
        """Detect citations and figure/table mentions within one chunk's text.

        Must run only after every chunk in the paper has been built --
        resolving a citation or mention requires `context.entity_chunk_ids`
        to be fully populated.

        Args:
            chunk: A paragraph-derived chunk to scan for references.
            context: Shared build context, with all chunks already registered.

        Returns:
            `CITES` and `REFERENCES` relationships detected in this chunk's text.
        """
        relationships: list[Relationship] = []
        seen_targets: set[ChunkId] = set()

        def _emit(
            number: int, lookup: Mapping[int, UUID], relationship_type: RelationshipType
        ) -> None:
            entity_id = lookup.get(number)
            if entity_id is None:
                return
            target_chunk_id = context.entity_chunk_ids.get(entity_id)
            if target_chunk_id is None or target_chunk_id in seen_targets:
                return
            seen_targets.add(target_chunk_id)
            relationships.append(
                Relationship(
                    paper_id=context.paper.id,
                    source_chunk_id=chunk.id,
                    target_chunk_id=target_chunk_id,
                    relationship_type=relationship_type,
                )
            )

        for match in _CITATION_PATTERN.finditer(chunk.text):
            _emit(int(match.group(1)), context.reference_number_lookup, RelationshipType.CITES)
        for match in _FIGURE_MENTION_PATTERN.finditer(chunk.text):
            number = parse_printed_number(match.group(1).upper())
            if number is not None:
                _emit(number, context.figure_number_lookup, RelationshipType.REFERENCES)
        for match in _TABLE_MENTION_PATTERN.finditer(chunk.text):
            number = parse_printed_number(match.group(1).upper())
            if number is not None:
                _emit(number, context.table_number_lookup, RelationshipType.REFERENCES)

        return relationships
