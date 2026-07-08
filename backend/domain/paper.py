"""The Paper aggregate: a fully parsed scientific document, its
bibliographic metadata, and its authors.
"""

from datetime import date
from uuid import UUID

from pydantic import Field, model_validator

from backend.domain.base import DomainModel
from backend.domain.identifiers import AuthorId, MetadataId, PaperId, generate_id
from backend.domain.reference import Reference
from backend.domain.structure import Paragraph, Section
from backend.domain.visuals import Caption, Figure, Table


class Author(DomainModel):
    """A person credited as an author of the paper.

    Attributes:
        id: Unique identifier for this author record.
        name: Author's full name as printed on the paper.
        affiliation: Author's institutional affiliation, if stated.
        email: Author's contact email, if stated.
        orcid: Author's ORCID identifier, if stated.
        order: Zero-based position of this author in the paper's author
            list, in the order they are listed on the paper.
    """

    id: AuthorId = Field(default_factory=lambda: AuthorId(generate_id()))
    name: str = Field(min_length=1)
    affiliation: str | None = None
    email: str | None = None
    orcid: str | None = None
    order: int = Field(ge=0)


class Metadata(DomainModel):
    """Bibliographic identity of a paper: how it is cited and discovered.

    Kept separate from `Paper`'s structural content (sections, figures,
    tables) because it serves a different set of consumers: a catalog or a
    search-by-author/year feature only ever needs `Metadata`, never the
    full parsed document body.

    Attributes:
        id: Unique identifier for this metadata record.
        title: Paper title.
        authors: Paper's authors, in listed order. Embedded here (rather
            than a flat, paper-referencing collection) because authors have
            no existence or reuse outside a paper's metadata.
        abstract: Paper's abstract text, if extracted.
        keywords: Author-supplied or extracted keywords.
        publication_date: Date the paper was published, if known.
        venue: Journal, conference, or preprint server the paper appeared in.
        doi: Digital Object Identifier, if the paper has one.
        arxiv_id: arXiv identifier, if the paper was posted to arXiv.
        language: ISO 639-1 language code of the paper's text.
        source_filename: Original filename of the uploaded PDF.
        page_count: Number of pages in the source PDF, if known.
    """

    id: MetadataId = Field(default_factory=lambda: MetadataId(generate_id()))
    title: str = Field(min_length=1)
    authors: list[Author] = Field(default_factory=list)
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    publication_date: date | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    language: str = "en"
    source_filename: str = Field(min_length=1)
    page_count: int | None = Field(default=None, ge=1)


class Paper(DomainModel):
    """A fully parsed scientific document: the aggregate root of the domain model.

    `Paper` is the output of the parser for one successfully parsed PDF. It
    does not track upload or parsing progress -- an in-progress or failed
    parse is not yet a `Paper`; that lifecycle belongs to a future
    upload/job concept in Module 3.

    All structural content (sections, paragraphs, figures, tables,
    captions, references) is stored as flat, paper-scoped collections
    rather than nested inside one another, because other entities need to
    reference them independently by id (e.g. a `Paragraph` references its
    `section_id`; a `Chunk` references the elements it was built from).
    Nesting them would force consumers to search a tree to resolve a
    reference they already hold by id.

    Attributes:
        id: Unique identifier for this paper.
        metadata: Bibliographic identity of the paper.
        sections: All sections in the paper, flat (see
            `Section.parent_section_id` for hierarchy), in no particular
            collection order -- consumers sort by `Section.order` within a
            parent.
        paragraphs: All body paragraphs in the paper.
        figures: All figures in the paper.
        tables: All tables in the paper.
        captions: All captions in the paper.
        references: All bibliography entries in the paper.
    """

    id: PaperId = Field(default_factory=lambda: PaperId(generate_id()))
    metadata: Metadata
    sections: list[Section] = Field(default_factory=list)
    paragraphs: list[Paragraph] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    captions: list[Caption] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_child_ownership(self) -> "Paper":
        for section in self.sections:
            self._require_owned_by_paper(section.id, section.paper_id)
        for paragraph in self.paragraphs:
            self._require_owned_by_paper(paragraph.id, paragraph.paper_id)
        for figure in self.figures:
            self._require_owned_by_paper(figure.id, figure.paper_id)
        for table in self.tables:
            self._require_owned_by_paper(table.id, table.paper_id)
        for caption in self.captions:
            self._require_owned_by_paper(caption.id, caption.paper_id)
        for reference in self.references:
            self._require_owned_by_paper(reference.id, reference.paper_id)
        return self

    def _require_owned_by_paper(self, entity_id: UUID, entity_paper_id: PaperId) -> None:
        if entity_paper_id != self.id:
            raise ValueError(
                f"entity {entity_id} has paper_id {entity_paper_id}, but was "
                f"attached to paper {self.id}"
            )
