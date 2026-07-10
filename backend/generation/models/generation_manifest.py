"""GenerationManifest: describes one complete generation run, for reproducibility."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import PaperId
from backend.generation.models.answer_status import AnswerStatus


class GenerationStatistics(BaseModel):
    """Summary counts and timing for one generation run.

    Attributes:
        context_sections_used: Number of evidence sections kept after
            Context Optimization.
        context_sections_dropped: Number of evidence sections removed
            (redundant, merged, or cut for budget) by Context Optimization.
        claims_total: Total claims extracted from the generated answer.
        claims_grounded: Claims that passed Grounding Validation.
        citations_resolved: Citation labels successfully resolved to real evidence.
        citations_unresolved: Citation labels that failed to resolve.
        prompt_tokens: Tokens consumed by the prompt, as reported by the provider.
        completion_tokens: Tokens generated for the answer, as reported by the provider.
        duration_ms: Total wall-clock time for the full generation call.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    context_sections_used: int = Field(ge=0)
    context_sections_dropped: int = Field(ge=0)
    claims_total: int = Field(ge=0)
    claims_grounded: int = Field(ge=0)
    citations_resolved: int = Field(ge=0)
    citations_unresolved: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    duration_ms: float = Field(ge=0)


class GenerationManifest(BaseModel):
    """Describes one complete generation run for a document and query.

    Persisted at `data/generation/<document_id>/generation_manifest.json`
    -- like `RetrievalManifest`, this is a reproducibility record of the
    most recent run, not a staleness cache-check target: generation is a
    per-query operation, never skipped as "already answered."

    Attributes:
        document_id: Identifier of the document queried.
        query: The original question, verbatim.
        generation_version: Schema version of this persisted manifest shape.
        prompt_version: Version of the prompt template and grounding rules used.
        provider: Name of the generation backend used.
        model_name: Model identifier as the provider understands it.
        model_version: Resolved, concrete model revision (e.g. an Ollama
            digest) -- never a floating tag, so "same model_version" is a
            guarantee of identical weights.
        answer_status: The deterministic evidence-sufficiency outcome for this answer.
        confidence: The deterministically computed confidence score.
        statistics: Summary counts and timing for this run.
        created_at: Timestamp this manifest was generated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: PaperId
    query: str = Field(min_length=1)
    generation_version: str
    prompt_version: str
    provider: str
    model_name: str
    model_version: str
    answer_status: AnswerStatus
    confidence: float = Field(ge=0.0, le=1.0)
    statistics: GenerationStatistics
    created_at: datetime
