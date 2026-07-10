"""Phase 5: Prompt Validation.

Runs after Prompt Composition and before any provider call. Fails loudly
on structural defects rather than letting a malformed prompt reach the
LLM -- catching a mistake here is far cheaper than discovering it only
after paying for a generation call.
"""

import re

from backend.generation.exceptions import (
    CitationPlaceholderError,
    ContextIntegrityError,
    DuplicatePromptEvidenceError,
    MissingEvidenceError,
    TokenBudgetExceededError,
)
from backend.generation.models.answer_plan import AnswerPlan
from backend.generation.models.generation_config import GenerationConfig
from backend.generation.models.prompt_context import PromptContext

_CHARS_PER_TOKEN_ESTIMATE = 4
_CITATION_LABEL_PATTERN = re.compile(r"^KU\d+$")


class PromptValidator:
    """Validates a composed prompt before it is sent to any provider."""

    def validate(self, prompt_context: PromptContext, config: GenerationConfig) -> None:
        """Validate a composed prompt.

        Args:
            prompt_context: The composed prompt to validate.
            config: Generation configuration the prompt will be sent under.

        Raises:
            TokenBudgetExceededError: The prompt's estimated size exceeds
                the model's context window.
            MissingEvidenceError: The answer plan requires citations but
                no evidence was provided.
            DuplicatePromptEvidenceError: The same knowledge unit appears
                more than once in the prompt's context.
            CitationPlaceholderError: A citation label is malformed or duplicated.
            ContextIntegrityError: A grounding or formatting rule is blank.
        """
        self._validate_token_budget(prompt_context, config)
        self._validate_missing_evidence(prompt_context, prompt_context.answer_plan)
        self._validate_duplicate_evidence(prompt_context)
        self._validate_citation_placeholders(prompt_context)
        self._validate_context_integrity(prompt_context)

    def _validate_token_budget(
        self, prompt_context: PromptContext, config: GenerationConfig
    ) -> None:
        estimated = _estimate_prompt_tokens(prompt_context)
        available = config.context_window - config.max_tokens
        if estimated > available:
            raise TokenBudgetExceededError(estimated_tokens=estimated, context_window=available)

    def _validate_missing_evidence(self, prompt_context: PromptContext, plan: AnswerPlan) -> None:
        if plan.requires_citations and not prompt_context.context_sections:
            raise MissingEvidenceError(
                reason="answer plan requires citations but no evidence was provided"
            )

    def _validate_duplicate_evidence(self, prompt_context: PromptContext) -> None:
        seen: set[str] = set()
        for section in prompt_context.context_sections:
            if section.knowledge_unit_id in seen:
                raise DuplicatePromptEvidenceError(knowledge_unit_id=section.knowledge_unit_id)
            seen.add(section.knowledge_unit_id)

    def _validate_citation_placeholders(self, prompt_context: PromptContext) -> None:
        seen_labels: set[str] = set()
        for section in prompt_context.context_sections:
            if not _CITATION_LABEL_PATTERN.match(section.citation_label):
                raise CitationPlaceholderError(
                    reason=f"citation label '{section.citation_label}' does not match the "
                    f"expected format (e.g. 'KU1')"
                )
            if section.citation_label in seen_labels:
                raise CitationPlaceholderError(
                    reason=f"citation label '{section.citation_label}' is used more than once"
                )
            seen_labels.add(section.citation_label)

    def _validate_context_integrity(self, prompt_context: PromptContext) -> None:
        for rule in (*prompt_context.grounding_rules, *prompt_context.formatting_rules):
            if not rule.strip():
                raise ContextIntegrityError(reason="a grounding or formatting rule is blank")
        if not prompt_context.system_prompt.strip():
            raise ContextIntegrityError(reason="system prompt is blank")
        if not prompt_context.user_question.strip():
            raise ContextIntegrityError(reason="user question is blank")


def _estimate_prompt_tokens(prompt_context: PromptContext) -> int:
    """Approximate the full rendered prompt's token cost.

    A char-count heuristic, not a real tokenizer (see
    `context/context_optimizer.py` for the same documented approximation)
    -- this check exists to catch cases that phase's own budget accounting
    missed (grounding/formatting rule text it doesn't account for), not to
    replace a real tokenizer.
    """
    total_chars = len(prompt_context.system_prompt) + len(prompt_context.user_question)
    total_chars += sum(len(rule) for rule in prompt_context.grounding_rules)
    total_chars += sum(len(rule) for rule in prompt_context.formatting_rules)
    total_chars += sum(len(section.text) for section in prompt_context.context_sections)
    return max(total_chars // _CHARS_PER_TOKEN_ESTIMATE, 1)
