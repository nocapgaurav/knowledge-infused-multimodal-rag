"""Maps a provider-agnostic `ExtractedDocument` into the domain `Paper`.

All semantic interpretation of document structure lives here: reconstructing
section hierarchy from a flat sequence of headings, recognizing the
abstract and references sections, and splitting a references block into
individual entries. A `DocumentParser` provider only ever hands over
structural facts (this text has this role, at this position); deciding
what those facts *mean* for our domain is this module's job.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from uuid import uuid4

from backend.domain import (
    BoundingBox,
    Caption,
    CaptionSubjectType,
    Figure,
    FigureId,
    Metadata,
    Paper,
    PaperId,
    Paragraph,
    Reference,
    Section,
    SectionId,
    Table,
    TableCell,
)
from backend.parser.exceptions import MissingRequiredMetadataError
from backend.parser.interfaces.extracted_document import (
    ExtractedBoundingBox,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    ExtractedTextBlock,
    ExtractedTextRole,
)

_SECTION_NUMBER_PREFIX = re.compile(r"^\s*\d+(\.\d+)*\.?\s*")
_BRACKET_NUMBER = re.compile(r"\[\d+\]")
_SPLIT_BEFORE_BRACKET_NUMBER = re.compile(r"(?=\[\d+\])")


def _normalize_heading(title: str) -> str:
    return _SECTION_NUMBER_PREFIX.sub("", title).strip().rstrip(":").strip().lower()


def _is_abstract_heading(title: str) -> bool:
    return _normalize_heading(title) == "abstract"


def _is_references_heading(title: str) -> bool:
    return _normalize_heading(title) in {"references", "bibliography"}


def _split_references(raw_blocks: list[str]) -> list[str]:
    """Split accumulated references-section text into individual entries.

    Splits on a `[n]`-style bracket-number marker when at least two are
    present (the dominant numbered-citation style); otherwise falls back to
    treating each accumulated text block as one entry. This is a bounded
    heuristic, not a guarantee -- unusual or inconsistent citation
    formatting may not split correctly.
    """
    combined = " ".join(text.strip() for text in raw_blocks if text.strip())
    if not combined:
        return []
    if len(_BRACKET_NUMBER.findall(combined)) >= 2:
        return [
            part.strip() for part in _SPLIT_BEFORE_BRACKET_NUMBER.split(combined) if part.strip()
        ]
    return [text.strip() for text in raw_blocks if text.strip()]


def _map_bounding_boxes(boxes: tuple[ExtractedBoundingBox, ...]) -> list[BoundingBox]:
    return [
        BoundingBox(page_number=box.page_number, x0=box.x0, y0=box.y0, x1=box.x1, y1=box.y1)
        for box in boxes
    ]


@dataclass
class _MappingState:
    """Mutable state accumulated while walking one document's content stream."""

    paper_id: PaperId
    section_stack: list[tuple[int, SectionId]] = field(default_factory=list)
    section_order: dict[SectionId | None, int] = field(default_factory=lambda: defaultdict(int))
    paragraph_order: dict[SectionId | None, int] = field(default_factory=lambda: defaultdict(int))
    figure_order: int = 0
    table_order: int = 0

    sections: list[Section] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    captions: list[Caption] = field(default_factory=list)
    figure_images: dict[FigureId, bytes] = field(default_factory=dict)

    special_mode: str | None = None  # "abstract" | "references" | None
    special_mode_level: int = 0
    abstract_blocks: list[str] = field(default_factory=list)
    reference_blocks: list[str] = field(default_factory=list)

    @property
    def current_section_id(self) -> SectionId | None:
        return self.section_stack[-1][1] if self.section_stack else None

    def resolve_parent(self, level: int) -> SectionId | None:
        while self.section_stack and self.section_stack[-1][0] >= level:
            self.section_stack.pop()
        return self.section_stack[-1][1] if self.section_stack else None


@dataclass(frozen=True)
class MappingResult:
    """Result of mapping an `ExtractedDocument` into the domain model.

    Attributes:
        paper: The fully constructed `Paper`.
        figure_images: Rendered image bytes for each figure that had one,
            keyed by the figure's id. Kept separate from `paper` because
            domain objects never carry binary payloads (see
            `Figure.asset_uri`'s docstring) -- the service persists these
            bytes at the path `asset_uri` already points to.
    """

    paper: Paper
    figure_images: dict[FigureId, bytes]


class DomainMapper:
    """Converts an `ExtractedDocument` into a domain `Paper`."""

    def to_paper(
        self, *, document_id: PaperId, source_filename: str, extracted: ExtractedDocument
    ) -> MappingResult:
        """Build a `Paper` from a parser's extracted content.

        Args:
            document_id: Identifier already reserved for this document
                during ingestion. Becomes `Paper.id`.
            source_filename: Original uploaded filename, for `Metadata`.
            extracted: Provider-agnostic parsed content to map.

        Returns:
            The constructed `Paper` together with its figures' image bytes.

        Raises:
            MissingRequiredMetadataError: No usable title could be determined.
        """
        if not extracted.title or not extracted.title.strip():
            raise MissingRequiredMetadataError(document_id=document_id, field="title")

        state = _MappingState(paper_id=document_id)
        for item in extracted.content:
            if isinstance(item, ExtractedTextBlock):
                self._handle_text_block(state, item)
            elif isinstance(item, ExtractedTable):
                self._handle_table(state, item)
            elif isinstance(item, ExtractedFigure):
                self._handle_figure(state, item)

        references = [
            Reference(paper_id=document_id, order=order, raw_text=raw_text)
            for order, raw_text in enumerate(_split_references(state.reference_blocks))
        ]
        abstract = "\n\n".join(state.abstract_blocks).strip() or None

        metadata = Metadata(
            title=extracted.title.strip(),
            abstract=abstract,
            source_filename=source_filename,
            page_count=extracted.page_count,
        )

        paper = Paper(
            id=document_id,
            metadata=metadata,
            sections=state.sections,
            paragraphs=state.paragraphs,
            figures=state.figures,
            tables=state.tables,
            captions=state.captions,
            references=references,
        )
        return MappingResult(paper=paper, figure_images=state.figure_images)

    def _handle_text_block(self, state: _MappingState, block: ExtractedTextBlock) -> None:
        if block.role is ExtractedTextRole.SECTION_HEADER:
            self._handle_heading(state, block)
            return
        self._handle_paragraph(state, block)

    def _handle_heading(self, state: _MappingState, block: ExtractedTextBlock) -> None:
        level = block.level or 1

        if state.special_mode is not None:
            if level <= state.special_mode_level:
                state.special_mode = None
            else:
                return  # a stray sub-heading inside an excluded special section

        if _is_abstract_heading(block.text) or _is_references_heading(block.text):
            state.special_mode = "abstract" if _is_abstract_heading(block.text) else "references"
            state.special_mode_level = level
            return

        parent_id = state.resolve_parent(level)
        order = state.section_order[parent_id]
        state.section_order[parent_id] += 1
        section = Section(
            paper_id=state.paper_id,
            parent_section_id=parent_id,
            title=block.text,
            level=level,
            order=order,
            bounding_boxes=_map_bounding_boxes(block.bounding_boxes),
        )
        state.sections.append(section)
        state.section_stack.append((level, section.id))

    def _handle_paragraph(self, state: _MappingState, block: ExtractedTextBlock) -> None:
        if state.special_mode == "abstract":
            state.abstract_blocks.append(block.text)
            return
        if state.special_mode == "references":
            state.reference_blocks.append(block.text)
            return

        section_id = state.current_section_id
        order = state.paragraph_order[section_id]
        state.paragraph_order[section_id] += 1
        state.paragraphs.append(
            Paragraph(
                paper_id=state.paper_id,
                section_id=section_id,
                order=order,
                text=block.text,
                bounding_boxes=_map_bounding_boxes(block.bounding_boxes),
            )
        )

    def _handle_table(self, state: _MappingState, extracted_table: ExtractedTable) -> None:
        table = Table(
            paper_id=state.paper_id,
            section_id=state.current_section_id,
            order=state.table_order,
            num_rows=extracted_table.num_rows,
            num_columns=extracted_table.num_columns,
            cells=[
                TableCell(
                    row=cell.row,
                    column=cell.column,
                    text=cell.text,
                    row_span=cell.row_span,
                    column_span=cell.column_span,
                    is_header=cell.is_header,
                )
                for cell in extracted_table.cells
            ],
            markdown=extracted_table.markdown,
            bounding_boxes=_map_bounding_boxes(extracted_table.bounding_boxes),
        )
        state.table_order += 1
        state.tables.append(table)
        if extracted_table.caption_text:
            state.captions.append(
                Caption(
                    paper_id=state.paper_id,
                    subject_type=CaptionSubjectType.TABLE,
                    subject_id=table.id,
                    text=extracted_table.caption_text,
                    bounding_boxes=_map_bounding_boxes(extracted_table.caption_bounding_boxes),
                )
            )

    def _handle_figure(self, state: _MappingState, extracted_figure: ExtractedFigure) -> None:
        figure_id = FigureId(uuid4())
        has_image = bool(extracted_figure.image_bytes and extracted_figure.image_format)
        asset_uri = f"figures/{figure_id}.{extracted_figure.image_format}" if has_image else None

        figure = Figure(
            id=figure_id,
            paper_id=state.paper_id,
            section_id=state.current_section_id,
            order=state.figure_order,
            asset_uri=asset_uri,
            bounding_boxes=_map_bounding_boxes(extracted_figure.bounding_boxes),
        )
        state.figure_order += 1
        state.figures.append(figure)
        if has_image and extracted_figure.image_bytes is not None:
            state.figure_images[figure_id] = extracted_figure.image_bytes
        if extracted_figure.caption_text:
            state.captions.append(
                Caption(
                    paper_id=state.paper_id,
                    subject_type=CaptionSubjectType.FIGURE,
                    subject_id=figure.id,
                    text=extracted_figure.caption_text,
                    bounding_boxes=_map_bounding_boxes(extracted_figure.caption_bounding_boxes),
                )
            )
