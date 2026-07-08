"""Query, Evidence, and Answer: the question-answering interaction.

Distinct from the document sub-domain (`Paper` and its structural
children): these entities represent a transaction between a user and the
system, not facts extracted from a paper.
"""

from datetime import UTC, datetime

from pydantic import Field

from backend.domain.base import DomainModel
from backend.domain.identifiers import (
    AnswerId,
    ChunkId,
    EvidenceId,
    PaperId,
    QueryId,
    generate_id,
)


class Query(DomainModel):
    """A question submitted by a user.

    Attributes:
        id: Unique identifier for this query.
        text: Raw question text.
        paper_ids: Papers this query is scoped to. An empty list means the
            query is unscoped and should be answered against the entire
            corpus -- this is what allows multi-document reasoning without
            a separate "corpus query" concept.
        created_at: Timestamp the query was submitted.
    """

    id: QueryId = Field(default_factory=lambda: QueryId(generate_id()))
    text: str = Field(min_length=1)
    paper_ids: list[PaperId] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Evidence(DomainModel):
    """A pointer from part of an answer back to the exact chunk it was
    grounded on.

    References a `Chunk` by id rather than a raw paragraph, figure, or
    table, because generation grounds on retrieved chunks -- provenance
    from a chunk back to the original paper element it was built from is
    `Chunk.source_element_ids`'s responsibility, not `Evidence`'s.

    Attributes:
        id: Unique identifier for this evidence record.
        paper_id: Identifier of the paper the cited chunk came from.
            Denormalized from the chunk (rather than requiring a join) so
            evidence can be listed, grouped, or filtered by source paper on
            its own -- which matters once an answer draws on multiple papers.
        chunk_id: Identifier of the chunk this evidence points to.
        quoted_text: Exact excerpt from the chunk that supports the
            answer. Kept separate from the full chunk text so the frontend
            can highlight the precise supporting span rather than an
            entire (possibly long) chunk, and so the excerpt can be checked
            against the chunk's actual text as a grounding sanity check.
        relevance_score: Retrieval similarity score for the chunk, if
            available. Semantics (range, metric) depend on the retrieval
            method that produced it; the domain layer does not interpret it.
    """

    id: EvidenceId = Field(default_factory=lambda: EvidenceId(generate_id()))
    paper_id: PaperId
    chunk_id: ChunkId
    quoted_text: str = Field(min_length=1)
    relevance_score: float | None = None


class Answer(DomainModel):
    """The system's generated response to a query.

    `evidence` requires at least one entry by construction. This is a
    deliberate type-level enforcement of the project's central premise --
    every answer must be evidence-grounded -- rather than a convention the
    generation module is merely expected to follow.

    Attributes:
        id: Unique identifier for this answer.
        query_id: Identifier of the query this answer responds to.
        text: Generated answer text.
        evidence: Evidence supporting this answer. Embedded directly
            (rather than referenced by id) because evidence has no
            existence or reuse outside the answer it supports, unlike the
            paper elements it ultimately points to.
        created_at: Timestamp the answer was generated.
    """

    id: AnswerId = Field(default_factory=lambda: AnswerId(generate_id()))
    query_id: QueryId
    text: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
