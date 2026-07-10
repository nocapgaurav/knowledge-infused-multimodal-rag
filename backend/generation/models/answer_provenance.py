"""AnswerProvenance: the full upstream chain of custody for one generated answer.

Distinct from `GenerationTrace` (what happened *during* generation): this
is what artifact versions the answer is ultimately grounded in, all the
way back through retrieval, the graph, embeddings, and the knowledge
representation -- continuing the same version-chain pattern
`RetrievalManifest` established for Module 9's own upstream (Modules 6-8).
"""

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import PaperId


class AnswerProvenance(BaseModel):
    """The complete upstream version chain and evidence usage for one answer.

    Attributes:
        document_id: Identifier of the document the answer was generated for.
        retrieval_version: Module 9's artifact schema version, from the
            source `EvidenceBundle`'s manifest.
        retrieval_strategy_version: Module 9's construction-rules version,
            from the source `EvidenceBundle`'s manifest.
        representation_version: Module 5's knowledge representation
            version, threaded through unchanged from the `EvidenceBundle`.
        embedding_version: Module 6's embedding model revision, threaded
            through unchanged from the `EvidenceBundle`.
        graph_version: Module 8's graph construction-rules version,
            threaded through unchanged from the `EvidenceBundle`.
        knowledge_unit_ids: Every knowledge unit actually cited in the
            final answer -- the precise evidentiary basis for what was said.
        evidence_bundle_checksum: Content hash of the source
            `EvidenceBundle`, so this answer's grounding can be verified
            against the exact evidence it was built from, not just "some"
            evidence for this document.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: PaperId
    retrieval_version: str
    retrieval_strategy_version: str
    representation_version: str
    embedding_version: str
    graph_version: str
    knowledge_unit_ids: tuple[str, ...] = Field(default_factory=tuple)
    evidence_bundle_checksum: str
