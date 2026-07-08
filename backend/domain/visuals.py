"""Figure, Table, and Caption: non-flowing-text content of a paper.

Grouped together because a `Caption` always describes exactly one `Figure`
or `Table`, and reading them side by side makes that relationship explicit.
"""

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from backend.domain.base import DomainModel
from backend.domain.identifiers import (
    CaptionId,
    FigureId,
    PaperId,
    SectionId,
    TableId,
    generate_id,
)
from backend.domain.value_objects import BoundingBox


class Figure(DomainModel):
    """An extracted figure (image, chart, or diagram) from a paper.

    Attributes:
        id: Unique identifier for this figure.
        paper_id: Identifier of the paper this figure belongs to.
        section_id: Identifier of the section this figure appears in, if any.
        order: Zero-based position of this figure among all figures in the
            paper, in document reading order.
        label: Figure's printed label (e.g. "Figure 3"), if extracted. Used
            to resolve in-text references such as "as shown in Figure 3".
        asset_uri: Opaque reference to the extracted image asset (e.g. a
            local path or object-storage URI), resolved by the storage
            module. The domain layer does not interpret this value.
        bounding_boxes: Location(s) of this figure in the source PDF, for
            frontend highlighting.
    """

    id: FigureId = Field(default_factory=lambda: FigureId(generate_id()))
    paper_id: PaperId
    section_id: SectionId | None = None
    order: int = Field(ge=0)
    label: str | None = None
    asset_uri: str | None = None
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)


class TableCell(DomainModel):
    """A single cell within a table.

    `TableCell` is a value object, not a domain entity: it has no
    independent identity or existence outside the `Table` that contains it,
    so it has no id of its own.

    Attributes:
        row: Zero-based row index of this cell.
        column: Zero-based column index of this cell.
        text: Cell's text content. May be empty -- unlike a paragraph, a
            blank table cell is a normal, meaningful occurrence.
        row_span: Number of rows this cell spans.
        column_span: Number of columns this cell spans.
        is_header: Whether this cell is a header cell.
    """

    row: int = Field(ge=0)
    column: int = Field(ge=0)
    text: str = ""
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    is_header: bool = False


class Table(DomainModel):
    """An extracted table from a paper.

    Cell-level structure (`cells`) is kept as the canonical representation
    because it is what table-aware reasoning ultimately needs; `markdown`
    is a denormalized text rendering kept alongside it purely for
    convenience -- feeding a table to a text embedding model or an LLM
    prompt as markdown is simpler than re-deriving it from `cells` every
    time.

    Attributes:
        id: Unique identifier for this table.
        paper_id: Identifier of the paper this table belongs to.
        section_id: Identifier of the section this table appears in, if any.
        order: Zero-based position of this table among all tables in the
            paper, in document reading order.
        label: Table's printed label (e.g. "Table 2"), if extracted.
        num_rows: Number of rows in the table.
        num_columns: Number of columns in the table.
        cells: Table's cells, in no particular collection order --
            consumers locate a cell by its `row`/`column`.
        markdown: Markdown rendering of the table, if produced by the parser.
        asset_uri: Opaque reference to a rendered image of the table, if
            one was produced (e.g. for visual/multimodal embedding).
        bounding_boxes: Location(s) of this table in the source PDF, for
            frontend highlighting.
    """

    id: TableId = Field(default_factory=lambda: TableId(generate_id()))
    paper_id: PaperId
    section_id: SectionId | None = None
    order: int = Field(ge=0)
    label: str | None = None
    num_rows: int = Field(ge=0)
    num_columns: int = Field(ge=0)
    cells: list[TableCell] = Field(default_factory=list)
    markdown: str | None = None
    asset_uri: str | None = None
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)


class CaptionSubjectType(StrEnum):
    """The kind of element a `Caption` describes."""

    FIGURE = "figure"
    TABLE = "table"


class Caption(DomainModel):
    """A caption describing exactly one figure or table.

    Modeled as its own entity, separate from `Figure`/`Table`, so that a
    caption's text and its position in the source PDF can be attributed and
    highlighted independently of the figure or table it describes.

    Points at its subject (`subject_type` + `subject_id`) rather than the
    figure/table pointing back at it, mirroring how a parser discovers a
    caption block and then determines what it captions -- and keeping the
    relationship declared in exactly one place.

    Attributes:
        id: Unique identifier for this caption.
        paper_id: Identifier of the paper this caption belongs to.
        subject_type: Whether this caption describes a figure or a table.
        subject_id: Identifier of the `Figure` or `Table` this caption
            describes, per `subject_type`.
        text: Caption's full text.
        bounding_boxes: Location(s) of this caption in the source PDF, for
            frontend highlighting.
    """

    id: CaptionId = Field(default_factory=lambda: CaptionId(generate_id()))
    paper_id: PaperId
    subject_type: CaptionSubjectType
    subject_id: UUID
    text: str = Field(min_length=1)
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
