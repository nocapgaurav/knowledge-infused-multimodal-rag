"""GenerationProvider: the port every concrete LLM backend implements.

Business logic (planner, context optimizer, prompt composer, validators,
quality assessor) depends only on this interface and on the
vendor-agnostic `PromptContext`/`GenerationConfig` models -- never on
Ollama, vLLM, llama.cpp, HuggingFace TGI, or an OpenAI-compatible API
directly. Swapping providers means writing one new file that renders the
same structured `PromptContext` into that backend's wire format; zero
changes anywhere else.
"""

from abc import ABC, abstractmethod

from backend.generation.models.generation_config import GenerationConfig
from backend.generation.models.prompt_context import PromptContext
from backend.generation.models.provider_response import ProviderResponse


class GenerationProvider(ABC):
    """An LLM backend capable of generating text from a structured prompt."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of this provider (e.g. "ollama")."""

    @abstractmethod
    def resolve_model_version(self, model: str) -> str:
        """Return a concrete, reproducible identifier for a model.

        Args:
            model: Model identifier as this provider understands it.

        Returns:
            A concrete revision identifier (e.g. a content digest) --
            never a floating "latest" label, so that "same model_version"
            is a guarantee of identical weights.

        Raises:
            GenerationProviderError: The model is not available, or its
                version could not be resolved.
        """

    @abstractmethod
    def generate(self, prompt_context: PromptContext, config: GenerationConfig) -> ProviderResponse:
        """Generate a response for a structured prompt.

        Args:
            prompt_context: The complete, structured prompt contents.
                Rendering this into the backend's own wire format (a
                message list, a single templated string, or anything
                else) is this method's responsibility, not the caller's.
            config: Generation parameters to use for this call.

        Returns:
            The raw generated response, with token/timing accounting.

        Raises:
            GenerationProviderError: The provider failed to produce a response.
        """
