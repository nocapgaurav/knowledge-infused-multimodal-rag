"""RetrievalTrace: a structured, reproducible record of every phase's decisions."""

from pydantic import BaseModel, ConfigDict, Field


class PhaseTrace(BaseModel):
    """What one phase did, for explainability and performance analysis.

    Attributes:
        phase: Name of the phase (e.g. "candidate_generation", "expansion",
            "evaluation", "assembly").
        input_count: Number of items the phase received.
        output_count: Number of items the phase produced.
        duration_ms: Wall-clock time the phase took.
        notes: Free-form, human-readable observations (e.g. budget limits
            hit, collection selected) -- deliberately not structured
            further, since the specific facts worth noting differ by phase.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: str = Field(min_length=1)
    input_count: int = Field(ge=0)
    output_count: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    notes: tuple[str, ...] = Field(default_factory=tuple)


class DroppedCandidate(BaseModel):
    """A candidate that was generated or expanded but excluded from the final bundle.

    Attributes:
        knowledge_unit_id: Identifier of the excluded candidate.
        phase: Name of the phase that excluded it.
        reason: Human-readable reason (e.g. "deduplicated: near-identical
            to <id>", "below evidence budget cutoff").
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    knowledge_unit_id: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class RetrievalTrace(BaseModel):
    """The complete, reproducible record of how one retrieval call was carried out.

    Attributes:
        phases: One entry per phase, in execution order.
        dropped: Every candidate excluded along the way, with why.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phases: tuple[PhaseTrace, ...] = Field(default_factory=tuple)
    dropped: tuple[DroppedCandidate, ...] = Field(default_factory=tuple)
