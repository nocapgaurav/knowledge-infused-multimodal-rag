"""Reference: a bibliography entry cited by a paper."""

from pydantic import Field

from backend.domain.base import DomainModel
from backend.domain.identifiers import PaperId, ReferenceId, generate_id


class Reference(DomainModel):
    """A single bibliography entry cited by a paper.

    A reference's authors are kept as plain strings rather than `Author`
    records: `Author` represents a credited author of *this* paper and is
    a first-class participant in this system, whereas a reference's authors
    are, by default, just descriptive text about an external work this
    system has not (yet) ingested.

    Attributes:
        id: Unique identifier for this reference.
        paper_id: Identifier of the paper that cites this reference.
        order: Zero-based position of this reference in the bibliography
            list, in document order (typically matching in-text numeric
            citation markers such as "[12]").
        raw_text: Full citation string as printed, always populated even
            if the structured fields below could not be parsed out of it.
        title: Cited work's title, if parsed.
        authors: Cited work's author names, if parsed.
        year: Cited work's publication year, if parsed.
        venue: Cited work's publication venue, if parsed.
        doi: Cited work's DOI, if parsed.
        url: URL for the cited work, if present (e.g. an arXiv link).
        resolved_paper_id: Identifier of the `Paper` this reference points
            to, if that cited work has itself been ingested into this
            system. `None` until such a resolution step is implemented;
            this field exists so that future cross-paper citation linking
            (multi-document reasoning) does not require a schema change.
    """

    id: ReferenceId = Field(default_factory=lambda: ReferenceId(generate_id()))
    paper_id: PaperId
    order: int = Field(ge=0)
    raw_text: str = Field(min_length=1)
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    resolved_paper_id: PaperId | None = None
