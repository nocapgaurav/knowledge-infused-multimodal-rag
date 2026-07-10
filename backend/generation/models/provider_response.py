"""ProviderResponse: the vendor-independent shape every GenerationProvider returns.

Deliberately minimal and provider-agnostic: raw generated text plus the
token/timing accounting needed for `GenerationStatistics` -- nothing here
exposes a specific backend's response shape (Ollama's `ChatResponse`, or
any future provider's own type).
"""

from pydantic import BaseModel, ConfigDict, Field


class ProviderResponse(BaseModel):
    """One generation call's raw output.

    Attributes:
        text: The generated answer text, unparsed.
        prompt_tokens: Tokens consumed by the prompt, as reported by the provider.
        completion_tokens: Tokens generated for the answer, as reported by the provider.
        duration_ms: Wall-clock time the provider took to generate this response.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
