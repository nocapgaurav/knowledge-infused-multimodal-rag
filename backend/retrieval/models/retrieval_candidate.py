"""RetrievalCandidate: one piece of potential evidence, with full discovery provenance."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import ChunkId, ChunkModality, PaperId, SectionId
from backend.retrieval.models.graph_path import GraphPath


class DiscoveryMethod(StrEnum):
    """How a candidate first entered the retrieval pool.

    "First" matters: a candidate reachable both by dense similarity and by
    graph traversal from another candidate keeps `DENSE_RETRIEVAL` -- Phase
    1 always runs before Phase 2, so the richer, direct signal is never
    discarded in favor of an indirect one.
    """

    DENSE_RETRIEVAL = "dense_retrieval"
    GRAPH_EXPANSION = "graph_expansion"


class RetrievalCandidate(BaseModel):
    """A single knowledge unit under consideration as evidence.

    Carries only structural and provenance facts -- no score. Scoring is
    Phase 3's job (`ScoredCandidate`); keeping this model score-free means
    the exact same object flows unmodified through Phase 1 and Phase 2.

    Attributes:
        knowledge_unit_id: Identifier of the underlying chunk.
        document_id: Identifier of the document this candidate belongs to.
        section_id: Identifier of the section this candidate belongs to, if any.
        modality: The kind of content this candidate represents.
        text: The candidate's text content.
        retrieval_context: The chunk's structural identity within the
            paper ("Title", "Figure 1", ...), when the chunking stage
            assigned one -- a deterministic ranking input, never content.
        page_numbers: Source PDF page(s) this chunk's content appears on,
            from its bounding boxes; empty when the parser recorded none.
        asset_uri: Opaque reference to a renderable image asset, if any.
        reading_order: Zero-based position among all chunks in the
            document, in reading order.
        citation_count: Number of other chunks that cite this one.
        dense_similarity: Cosine similarity to the query, if this candidate
            was found (or independently confirmed) by dense retrieval.
            `None` for a candidate discovered only through graph expansion.
        discovery_method: How this candidate first entered the pool.
        graph_path: The traversal that discovered this candidate. Empty
            (`depth == 0`) for a `DENSE_RETRIEVAL` candidate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    knowledge_unit_id: ChunkId
    document_id: PaperId
    section_id: SectionId | None
    modality: ChunkModality
    text: str = Field(min_length=1)
    retrieval_context: str | None = None
    page_numbers: tuple[int, ...] = ()
    asset_uri: str | None
    reading_order: int = Field(ge=0)
    citation_count: int = Field(ge=0)
    dense_similarity: float | None = None
    discovery_method: DiscoveryMethod
    graph_path: GraphPath = Field(default_factory=GraphPath)
