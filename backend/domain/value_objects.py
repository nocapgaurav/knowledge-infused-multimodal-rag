"""Value objects shared across multiple domain entities.

Unlike entities, value objects have no identity of their own -- two
`BoundingBox` instances with the same coordinates are interchangeable. They
exist only as attributes of an entity, never as independently addressable
records.
"""

from pydantic import Field, model_validator

from backend.domain.base import DomainModel


class BoundingBox(DomainModel):
    """A rectangular region on a single page of the source PDF.

    Used by every entity that the frontend must be able to highlight in the
    original document (paragraphs, figures, tables, captions, chunks, and
    section headings).

    The coordinate system and unit (e.g. PDF points vs. a normalized 0-1
    range) are defined by whichever parser produced the box; the domain
    layer treats them opaquely as "a rectangle on a page".

    Attributes:
        page_number: 1-indexed page this box is located on.
        x0: Left edge of the box.
        y0: Top edge of the box.
        x1: Right edge of the box.
        y1: Bottom edge of the box.
    """

    page_number: int = Field(ge=1)
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)

    @model_validator(mode="after")
    def _validate_rectangle(self) -> "BoundingBox":
        if self.x1 < self.x0:
            raise ValueError("x1 must be greater than or equal to x0")
        if self.y1 < self.y0:
            raise ValueError("y1 must be greater than or equal to y0")
        return self
