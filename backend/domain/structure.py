"""Section and Paragraph: the textual body structure of a paper."""

from pydantic import Field

from backend.domain.base import DomainModel
from backend.domain.identifiers import PaperId, ParagraphId, SectionId, generate_id
from backend.domain.value_objects import BoundingBox


class Section(DomainModel):
    """A titled section or subsection of a paper (e.g. "3.2 Experimental Setup").

    Attributes:
        id: Unique identifier for this section.
        paper_id: Identifier of the paper this section belongs to.
        parent_section_id: Identifier of the enclosing section, if this is
            a subsection. `None` for top-level sections.
        title: Section heading text.
        level: Heading depth; 1 for top-level sections, 2 for their direct
            subsections, and so on.
        order: Zero-based position of this section among sibling sections
            (those sharing the same `parent_section_id`), in document
            reading order.
        bounding_boxes: Location(s) of the section heading text in the
            source PDF, for frontend highlighting.
    """

    id: SectionId = Field(default_factory=lambda: SectionId(generate_id()))
    paper_id: PaperId
    parent_section_id: SectionId | None = None
    title: str = Field(min_length=1)
    level: int = Field(ge=1)
    order: int = Field(ge=0)
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)


class Paragraph(DomainModel):
    """A contiguous block of body text within a paper.

    Attributes:
        id: Unique identifier for this paragraph.
        paper_id: Identifier of the paper this paragraph belongs to.
        section_id: Identifier of the section this paragraph appears in.
            `None` if the paragraph could not be associated with a
            detected section (e.g. the abstract, or text before the first
            heading).
        order: Zero-based position of this paragraph among sibling
            paragraphs (those sharing the same `section_id`), in document
            reading order.
        text: Paragraph's full text content.
        bounding_boxes: Location(s) of this paragraph in the source PDF,
            for frontend highlighting and evidence attribution. A single
            paragraph may span multiple regions (e.g. across a column or
            page break), hence a list rather than one box.
    """

    id: ParagraphId = Field(default_factory=lambda: ParagraphId(generate_id()))
    paper_id: PaperId
    section_id: SectionId | None = None
    order: int = Field(ge=0)
    text: str = Field(min_length=1)
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
