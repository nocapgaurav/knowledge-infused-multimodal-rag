"""GroundedResponse: the sole, complete output of this module.

Every claim in `answer`/`executive_summary` is expected to trace to
`supporting_evidence`/`resolved_citations` -- that traceability is what
Grounding Validation and Citation Resolution exist to guarantee before
this object is ever constructed.
"""

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import ChunkModality, PaperId
from backend.generation.models.answer_provenance import AnswerProvenance
from backend.generation.models.answer_status import AnswerStatus
from backend.generation.models.citation import ResolvedCitation
from backend.generation.models.generation_manifest import GenerationStatistics
from backend.generation.models.generation_trace import GenerationTrace


class SupportingEvidenceItem(BaseModel):
    """One piece of evidence rendered for display in the response's
    Supporting Evidence section.

    Attributes:
        label: The citation label this evidence was shown to the model under.
        knowledge_unit_id: Identifier of the underlying knowledge unit.
        text: The evidence's text content.
        modality: The kind of content this evidence represents.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    knowledge_unit_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    modality: ChunkModality


class GroundedResponse(BaseModel):
    """The complete, evidence-grounded answer produced by one generation call.

    Attributes:
        document_id: Identifier of the document the question was asked against.
        query: The original question, verbatim.
        answer: The detailed answer body.
        executive_summary: A short, standalone summary of the answer.
        supporting_evidence: Evidence rendered for display, in citation order.
        resolved_citations: Every citation label used in the answer,
            resolved to real evidence.
        limitations: Explicit statements of what the evidence could not
            support -- always populated when `answer_status` is not
            `SUFFICIENT_EVIDENCE`.
        references: Formatted reference strings for display.
        warnings: Non-fatal issues surfaced during generation, curated for
            end-user display (the full detail lives in `generation_trace`).
        confidence: Deterministically computed confidence in `[0.0, 1.0]`.
        answer_status: The deterministic evidence-sufficiency outcome.
        generation_metadata: Small, explicitly-populated bag of additional
            run metadata not otherwise captured by a named field (e.g.
            `{"question_type": "comparative"}`).
        prompt_version: Version of the prompt template and grounding rules used.
        model_name: Model identifier as the provider understands it.
        model_version: Resolved, concrete model revision.
        generation_trace: The reproducible, phase-by-phase record of this run.
        generation_statistics: Summary counts and timing for this run.
        answer_provenance: The full upstream chain of custody for this answer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: PaperId
    query: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    executive_summary: str = Field(min_length=1)
    supporting_evidence: tuple[SupportingEvidenceItem, ...] = Field(default_factory=tuple)
    resolved_citations: tuple[ResolvedCitation, ...] = Field(default_factory=tuple)
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    references: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(ge=0.0, le=1.0)
    answer_status: AnswerStatus
    generation_metadata: dict[str, str] = Field(default_factory=dict)
    prompt_version: str
    model_name: str
    model_version: str
    generation_trace: GenerationTrace
    generation_statistics: GenerationStatistics
    answer_provenance: AnswerProvenance
