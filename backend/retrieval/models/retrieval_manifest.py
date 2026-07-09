"""RetrievalManifest: describes one complete retrieval run, for reproducibility."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import PaperId


class RetrievalStatistics(BaseModel):
    """Summary counts and timing for one retrieval run.

    Attributes:
        candidates_generated: Number of candidates found by dense retrieval (Phase 1).
        candidates_expanded: Number of additional candidates found by graph
            expansion (Phase 2).
        candidates_scored: Total candidates evaluated (Phase 3) --
            `candidates_generated + candidates_expanded`.
        evidence_groups: Number of evidence groups assembled (Phase 4).
        evidence_items: Total candidates included across all evidence
            groups (primary plus supporting).
        duration_ms: Total wall-clock time for the full retrieval call.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates_generated: int = Field(ge=0)
    candidates_expanded: int = Field(ge=0)
    candidates_scored: int = Field(ge=0)
    evidence_groups: int = Field(ge=0)
    evidence_items: int = Field(ge=0)
    duration_ms: float = Field(ge=0)


class RetrievalManifest(BaseModel):
    """Describes one complete retrieval run for a document and query.

    Persisted at `data/retrieval/<document_id>/retrieval_manifest.json` --
    unlike prior modules' manifests, this is not a staleness-check record
    (retrieval is a per-query, not per-document, operation and is never
    skipped as "already fresh") but a reproducibility record: everything
    needed to know exactly which upstream artifact versions and which
    version of this module's own strategy produced a given bundle.

    Attributes:
        document_id: Identifier of the document queried.
        query: The original question, verbatim.
        retrieval_version: Schema version of this persisted manifest shape.
        retrieval_strategy_version: Version of this module's own retrieval
            rules (candidate generation, expansion, ranking, assembly).
            Bumped when those rules change, independently of the manifest
            schema or any upstream artifact.
        representation_version: The knowledge representation version this
            run's embeddings were computed from (read from Module 6's
            embedding manifest -- never recomputed here).
        embedding_version: The embedding model revision used to index the
            vectors this run searched (read from Module 6's embedding manifest).
        graph_version: The graph construction rules version of the graph
            this run traversed (read from Module 8's graph manifest).
        statistics: Summary counts and timing for this run.
        created_at: Timestamp this manifest was generated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: PaperId
    query: str = Field(min_length=1)
    retrieval_version: str
    retrieval_strategy_version: str
    representation_version: str
    embedding_version: str
    graph_version: str
    statistics: RetrievalStatistics
    created_at: datetime
