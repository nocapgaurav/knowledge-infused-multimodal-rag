"""Ollama-based implementation of `GenerationProvider`.

This is the only file in the application permitted to import `ollama`.

Owns the entire translation from a vendor-agnostic `PromptContext` into
Ollama's `/api/chat` message-list shape (confirmed against a real Ollama
0.31 instance: `role`/`content` messages, `options.{temperature,top_p,
num_predict,num_ctx}`, and a response exposing `message.content`,
`prompt_eval_count`, `eval_count`, `total_duration` in nanoseconds) -- no
other file in this module ever sees that shape.
"""

import logging

from ollama import Client, RequestError, ResponseError

from backend.generation.exceptions import GenerationProviderError
from backend.generation.interfaces.generation_provider import GenerationProvider
from backend.generation.models.generation_config import GenerationConfig
from backend.generation.models.prompt_context import PromptContext
from backend.generation.models.provider_response import ProviderResponse

logger = logging.getLogger(__name__)

_NANOSECONDS_PER_MILLISECOND = 1_000_000


class OllamaProvider(GenerationProvider):
    """Generation provider backed by a local or remote Ollama instance."""

    def __init__(self, host: str, timeout_seconds: float = 120.0) -> None:
        """Connect to an Ollama instance.

        Args:
            host: Base URL of the Ollama server (e.g. "http://localhost:11434").
            timeout_seconds: Request timeout for generation calls. Generous
                by default -- local LLM inference on modest hardware can
                genuinely take tens of seconds for a full answer.
        """
        self._client = Client(host=host, timeout=timeout_seconds)

    @property
    def provider_name(self) -> str:
        return "ollama"

    def resolve_model_version(self, model: str) -> str:
        try:
            response = self._client.list()
        except (ResponseError, RequestError) as exc:
            raise GenerationProviderError(reason=str(exc)) from exc

        for entry in response.models:
            if entry.model == model and entry.digest:
                return entry.digest
        raise GenerationProviderError(
            reason=f"model '{model}' is not available on this Ollama instance"
        )

    def generate(self, prompt_context: PromptContext, config: GenerationConfig) -> ProviderResponse:
        messages = _render_messages(prompt_context)
        try:
            response = self._client.chat(
                model=config.model,
                messages=messages,
                stream=False,
                options={
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "num_predict": config.max_tokens,
                    "num_ctx": config.context_window,
                },
            )
        except (ResponseError, RequestError) as exc:
            raise GenerationProviderError(reason=str(exc)) from exc

        content = response.message.content
        if not content:
            raise GenerationProviderError(reason="provider returned an empty response")

        return ProviderResponse(
            text=content,
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
            duration_ms=(response.total_duration or 0) / _NANOSECONDS_PER_MILLISECOND,
        )


def _render_messages(prompt_context: PromptContext) -> list[dict[str, str]]:
    system_lines = [prompt_context.system_prompt, "", "Grounding rules:"]
    system_lines.extend(f"- {rule}" for rule in prompt_context.grounding_rules)
    system_lines.extend(["", "Formatting rules:"])
    system_lines.extend(f"- {rule}" for rule in prompt_context.formatting_rules)

    evidence_lines = ["Evidence:"]
    evidence_lines.extend(
        f"[{section.citation_label}] {section.text}" for section in prompt_context.context_sections
    )
    evidence_lines.extend(["", f"Question: {prompt_context.user_question}"])

    return [
        {"role": "system", "content": "\n".join(system_lines)},
        {"role": "user", "content": "\n".join(evidence_lines)},
    ]
