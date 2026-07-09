"""EvidenceGroup: a coherent cluster of evidence assembled around one primary candidate."""

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import ChunkModality
from backend.retrieval.models.ranking import ScoredCandidate


class EvidenceGroup(BaseModel):
    """One coherent unit of evidence: a primary finding plus its supporting context.

    Modeled after how a human reader actually uses scientific evidence --
    a paragraph is rarely read alone; its supporting figure, table, or
    cited reference is read alongside it. A group is that unit, assembled
    once so Module 10 never has to re-derive which candidates belong together.

    Attributes:
        group_id: Deterministic identifier for this group, derived from
            `primary.candidate.knowledge_unit_id` -- stable across
            identical reruns, never freshly generated.
        primary: The candidate this group is built around -- typically the
            highest-ranked candidate not yet claimed by another group.
        supporting: Other candidates assembled alongside the primary (its
            referenced figure/table, cited references, adjacent context).
            Each supporting candidate belongs to exactly one group; no
            candidate is ever duplicated across groups.
        modalities: Distinct content modalities present in this group
            (primary plus supporting), for diversity reporting.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: str = Field(min_length=1)
    primary: ScoredCandidate
    supporting: tuple[ScoredCandidate, ...] = Field(default_factory=tuple)
    modalities: tuple[ChunkModality, ...] = Field(min_length=1)
