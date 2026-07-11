"""Citation resolution results.

The model is only ever allowed to cite by the short label it was shown
(see `PromptContext.ContextSection.citation_label`); a citation only
becomes real once it is resolved here against that same label -> id
mapping. Nothing the model writes is trusted as a valid reference on its own.
"""

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import ChunkModality
from backend.domain.value_objects import BoundingBox


class ResolvedCitation(BaseModel):
    """A citation label successfully resolved to real evidence.

    Attributes:
        label: The citation label as it appeared in the generated answer.
        knowledge_unit_id: The real knowledge unit this label resolves to.
        text_excerpt: The cited evidence's own text, for display alongside
            the citation.
        display_label: Human-readable identity of the cited evidence
            ("Figure 2", "Section: III. Methodology", ...), when known --
            what a reader should see instead of the internal label.
        page_numbers: Source PDF page(s) the cited evidence appears on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    knowledge_unit_id: str = Field(min_length=1)
    text_excerpt: str = Field(min_length=1)
    display_label: str | None = None
    page_numbers: tuple[int, ...] = ()
    bounding_boxes: tuple[BoundingBox, ...] = ()
    modality: ChunkModality = ChunkModality.TEXT


class UnresolvedCitation(BaseModel):
    """A citation label the model used that does not resolve to any real evidence.

    Attributes:
        label: The citation label as it appeared in the generated answer.
        reason: Why it failed to resolve (e.g. "label was never shown to the model").
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class CitationResolutionReport(BaseModel):
    """The complete citation resolution outcome for a generated answer.

    Attributes:
        resolved: Every citation label that resolved to real evidence.
        unresolved: Every citation label that did not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    resolved: tuple[ResolvedCitation, ...] = Field(default_factory=tuple)
    unresolved: tuple[UnresolvedCitation, ...] = Field(default_factory=tuple)
