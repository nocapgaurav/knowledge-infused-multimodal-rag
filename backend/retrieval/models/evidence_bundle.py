"""EvidenceBundle: the sole, complete output of this module.

Only Module 10 (Grounded Answer Generation) may consume this. Deliberately
does not duplicate per-candidate provenance (graph paths, similarity
scores, ranking explanations) as separate flat top-level arrays -- every
one of those facts already lives on the relevant `RetrievalCandidate` or
`ScoredCandidate` inside `candidates`/`evidence_groups`. A parallel flat
array would be a second, driftable copy of the same data.
"""

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import PaperId
from backend.retrieval.models.evidence_group import EvidenceGroup
from backend.retrieval.models.retrieval_candidate import RetrievalCandidate
from backend.retrieval.models.retrieval_manifest import RetrievalManifest
from backend.retrieval.models.retrieval_trace import RetrievalTrace


class EvidenceBundle(BaseModel):
    """The complete evidence package produced by one retrieval call.

    Attributes:
        document_id: Identifier of the document queried.
        query: The original question, verbatim.
        candidates: Every candidate considered -- Phase 1's dense matches
            plus Phase 2's graph-expanded discoveries -- regardless of
            whether it was ultimately selected into an evidence group.
            Kept for full traceability of what was seen, not just what was used.
        evidence_groups: The final, assembled evidence, ranked best first.
        trace: The reproducible record of every phase's decisions.
        manifest: Versioning and statistics for this run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: PaperId
    query: str = Field(min_length=1)
    candidates: tuple[RetrievalCandidate, ...] = Field(default_factory=tuple)
    evidence_groups: tuple[EvidenceGroup, ...] = Field(default_factory=tuple)
    trace: RetrievalTrace
    manifest: RetrievalManifest
