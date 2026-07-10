"""GenerationTrace: a structured, reproducible record of every phase's decisions."""

from pydantic import BaseModel, ConfigDict, Field


class PhaseTrace(BaseModel):
    """What one phase did, for explainability and performance analysis.

    Attributes:
        phase: Name of the phase (e.g. "answer_planning", "context_optimization").
        input_count: Number of items the phase received.
        output_count: Number of items the phase produced.
        duration_ms: Wall-clock time the phase took.
        notes: Free-form, human-readable observations specific to this phase.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: str = Field(min_length=1)
    input_count: int = Field(ge=0)
    output_count: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    notes: tuple[str, ...] = Field(default_factory=tuple)


class GenerationTrace(BaseModel):
    """The complete, reproducible record of how one answer was generated.

    Attributes:
        phases: One entry per phase, in execution order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phases: tuple[PhaseTrace, ...] = Field(default_factory=tuple)
