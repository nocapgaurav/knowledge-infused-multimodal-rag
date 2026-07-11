"""Chunk: the retrieval-ready unit produced by the chunking stage."""

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from backend.domain.base import DomainModel
from backend.domain.identifiers import ChunkId, PaperId, SectionId, generate_id
from backend.domain.value_objects import BoundingBox


class ChunkModality(StrEnum):
    """The kind of content a chunk represents.

    Downstream modules use this to decide how to embed a chunk (a text
    embedding model for `TEXT`/`TABLE`, a multimodal/image embedding model
    for `FIGURE`) and how to render it in the UI.
    """

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class Chunk(DomainModel):
    """A retrieval-ready unit of content, produced by the chunking module
    from one or more paper elements.

    Deliberately not nested inside `Paper`: a chunk is produced later, by a
    different stage (chunking, not parsing), and its merge/split policy
    (how many paragraphs make up one chunk, whether a long paragraph is
    split across several) is that stage's business logic, not a fact fixed
    at parse time.

    A `Chunk` also does not embed the vector produced from it: the vector
    belongs in the vector database (Qdrant), keyed by this chunk's `id`.
    Storing it here would couple every consumer of the domain layer to a
    specific embedding model's dimensionality, and would duplicate data the
    vector database already owns.

    Attributes:
        id: Unique identifier for this chunk.
        paper_id: Identifier of the paper this chunk was derived from.
        section_id: Identifier of the section this chunk belongs to, if any.
        order: Zero-based position of this chunk among all chunks derived
            from the same paper, in document reading order. Adjacent chunks
            are contextual neighbors, useful for neighbor-expansion at
            retrieval time.
        modality: The kind of content this chunk represents.
        text: Text content to embed and display for this chunk (e.g. a
            paragraph's text, a table's markdown rendering, or a figure's
            caption text).
        asset_uri: Opaque reference to a renderable image asset for this
            chunk, if `modality` is `FIGURE` (or a rendered table image).
        token_count: Number of tokens in `text`, if computed by the
            chunking module. Recorded here to avoid re-tokenizing text
            later purely to check size.
        retrieval_context: Short structural identity of this chunk within
            the paper (e.g. "Title", "Abstract", "Reference [14]",
            "Figure 1", "Table 2", "Title page"), set by the chunking
            strategy that built it. Used to contextualize the text at
            embedding time and as a ranking signal at retrieval time --
            never shown as content, so `text` remains exactly what the
            reader sees (and what the frontend matches against the PDF).
            `None` for ordinary body text, which needs no qualifier.
        source_element_ids: Identifiers of the paragraph(s), figure(s),
            table(s), or caption(s) this chunk was built from. A chunk may
            combine several source elements (e.g. small paragraphs merged
            together) or share a source element with other chunks (e.g. one
            long paragraph split across chunks); which combinations are
            valid is the chunking module's concern, not the domain layer's.
        bounding_boxes: Location(s) of this chunk in the source PDF, for
            frontend highlighting -- typically the union of its source
            elements' bounding boxes.
    """

    id: ChunkId = Field(default_factory=lambda: ChunkId(generate_id()))
    paper_id: PaperId
    section_id: SectionId | None = None
    order: int = Field(ge=0)
    modality: ChunkModality
    text: str = Field(min_length=1)
    retrieval_context: str | None = None
    asset_uri: str | None = None
    token_count: int | None = Field(default=None, ge=0)
    source_element_ids: list[UUID] = Field(default_factory=list)
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
