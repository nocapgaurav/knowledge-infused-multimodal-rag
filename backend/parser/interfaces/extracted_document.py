"""Provider-agnostic intermediate representation of a parsed document.

This is the contract between a `DocumentParser` implementation (e.g. the
Docling-based provider) and the domain mapper. No parsing-library types
may appear on either side of this boundary -- that is what keeps the
mapper, validator, and the rest of the application unaware that Docling
exists.

Plain `dataclasses`, not Pydantic: this representation is trusted, internal
plumbing constructed and consumed entirely within one parse request, with
no external boundary to validate against, so Pydantic's validation
machinery would add cost without adding safety here.
"""

from dataclasses import dataclass
from enum import StrEnum


class ExtractedTextRole(StrEnum):
    """The structural role a parser assigned to a text block.

    Deliberately small: a source parser's finer-grained labels (footnote,
    list item, code, ...) all fold into `PARAGRAPH` here. The mapper does
    not need finer distinctions than "this is a heading" vs "this is body
    text" to reconstruct document structure.
    """

    SECTION_HEADER = "section_header"
    PARAGRAPH = "paragraph"


@dataclass(frozen=True, slots=True)
class ExtractedBoundingBox:
    """A rectangular region on a page, normalized to a top-left origin.

    Attributes:
        page_number: 1-indexed page this box is located on.
        x0: Left edge of the box.
        y0: Top edge of the box.
        x1: Right edge of the box.
        y1: Bottom edge of the box.
    """

    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, slots=True)
class ExtractedTextBlock:
    """A single block of text, with its structural role and position.

    Attributes:
        role: Structural role of this block.
        text: Text content of this block.
        level: Heading depth; only meaningful when `role` is `SECTION_HEADER`.
        bounding_boxes: Location(s) of this block in the source PDF.
    """

    role: ExtractedTextRole
    text: str
    level: int | None
    bounding_boxes: tuple[ExtractedBoundingBox, ...]


@dataclass(frozen=True, slots=True)
class ExtractedTableCell:
    """A single cell within an extracted table.

    Attributes:
        row: Zero-based row index of this cell.
        column: Zero-based column index of this cell.
        text: Cell's text content.
        row_span: Number of rows this cell spans.
        column_span: Number of columns this cell spans.
        is_header: Whether this cell is a header cell.
    """

    row: int
    column: int
    text: str
    row_span: int
    column_span: int
    is_header: bool


@dataclass(frozen=True, slots=True)
class ExtractedTable:
    """A single extracted table, with an optional caption.

    Attributes:
        cells: Table's cells, in no particular order.
        num_rows: Number of rows in the table.
        num_columns: Number of columns in the table.
        markdown: Markdown rendering of the table, if the parser can produce one.
        caption_text: Caption text describing this table, if found.
        caption_bounding_boxes: Location(s) of the caption text, if found.
        bounding_boxes: Location(s) of the table itself in the source PDF.
    """

    cells: tuple[ExtractedTableCell, ...]
    num_rows: int
    num_columns: int
    markdown: str | None
    caption_text: str | None
    caption_bounding_boxes: tuple[ExtractedBoundingBox, ...]
    bounding_boxes: tuple[ExtractedBoundingBox, ...]


@dataclass(frozen=True, slots=True)
class ExtractedFigure:
    """A single extracted figure, with an optional caption and rendered image.

    Attributes:
        caption_text: Caption text describing this figure, if found.
        caption_bounding_boxes: Location(s) of the caption text, if found.
        image_bytes: Rendered image content, if the parser could produce one.
        image_format: Image format of `image_bytes` (e.g. "png"), if present.
        bounding_boxes: Location(s) of the figure itself in the source PDF.
    """

    caption_text: str | None
    caption_bounding_boxes: tuple[ExtractedBoundingBox, ...]
    image_bytes: bytes | None
    image_format: str | None
    bounding_boxes: tuple[ExtractedBoundingBox, ...]


type ExtractedContentItem = ExtractedTextBlock | ExtractedTable | ExtractedFigure
"""One item in a document's reading-order content stream."""


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """The full, provider-agnostic result of parsing one PDF.

    Attributes:
        title: Document title, if the parser could identify one.
        page_count: Number of pages in the source PDF.
        content: All text blocks, tables, and figures, in true document
            reading order. Page headers, page footers, and already-resolved
            captions are excluded. Semantic interpretation of this stream
            (section hierarchy, abstract, references) is the mapper's job,
            not the parser's.
    """

    title: str | None
    page_count: int
    content: tuple[ExtractedContentItem, ...]
