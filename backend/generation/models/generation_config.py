"""GenerationConfig: vendor-independent configuration for one generation run.

Never hardcoded in business logic -- constructed once from application
settings and threaded through every phase that needs it (context
optimization needs `context_window`; the provider needs everything).
"""

from pydantic import BaseModel, ConfigDict, Field


class GenerationConfig(BaseModel):
    """Configuration for a single generation run, independent of any provider's wire format.

    Attributes:
        provider: Name of the generation backend (e.g. "ollama"). A plain
            string, not a closed enum -- adding a new provider must never
            require changing this model, only adding a new file under
            `providers/`.
        model: Model identifier as the provider understands it (e.g.
            "qwen2.5:7b-instruct"). Never hardcoded in code -- always
            supplied via configuration.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum tokens the provider may generate for the answer.
        context_window: Maximum total tokens (prompt + completion) the
            model supports -- Context Optimization sizes the evidence it
            keeps against this, never against a value it invents.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    max_tokens: int = Field(gt=0)
    context_window: int = Field(gt=0)
