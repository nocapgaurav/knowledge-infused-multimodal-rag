"""Phase 3: Context Optimization.

Turns Module 9's already-ranked, already-diversified evidence groups into
a token-budgeted list of `ContextSection`s. Module 9 guarantees no
candidate appears in more than one group; this phase adds the two things
only it can decide: how much evidence actually fits the model being used,
and whether any two knowledge units' text is redundant enough that
showing both wastes budget without adding information.

Never assumes a fixed token limit: the budget is derived from
`GenerationConfig.context_window` and `max_tokens`, both configurable, so
a larger or smaller model changes how much evidence is kept without any
code change here.
"""

from dataclasses import dataclass

from backend.generation.models.answer_plan import AnswerPlan
from backend.generation.models.generation_config import GenerationConfig
from backend.generation.models.prompt_context import ContextSection
from backend.retrieval.models import EvidenceBundle, ScoredCandidate

_CHARS_PER_TOKEN_ESTIMATE = 4
"""A widely-used, documented rule-of-thumb for English text (roughly what
OpenAI's own tokenizer guidance cites) -- an approximation, not a real
tokenizer. Good enough for a conservative budget check; integrating the
serving model's own tokenizer is a natural future improvement, not a
correctness requirement here (see the final report's known limitations)."""

_RESERVED_TOKENS_FOR_PROMPT_OVERHEAD = 512
"""Headroom reserved for the system prompt, grounding/formatting rules,
and the question itself -- everything in the prompt that isn't evidence."""


@dataclass(frozen=True)
class ContextOptimizationResult:
    """The outcome of one context optimization call.

    Attributes:
        context_sections: The evidence kept, in final presentation order,
            each already assigned its citation label.
        notes: Human-readable notes on what was dropped or merged, and why.
        estimated_prompt_tokens: Estimated token cost of the kept evidence
            alone (excluding prompt overhead), for reporting.
        total_candidates_considered: Total evidence items considered
            before any redundancy removal or budget trimming -- lets a
            caller report how much was dropped without recomputing the
            flattened candidate list itself.
    """

    context_sections: list[ContextSection]
    notes: list[str]
    estimated_prompt_tokens: int
    total_candidates_considered: int


class ContextOptimizer:
    """Selects and orders evidence to fit within a model's context window."""

    def optimize(
        self, bundle: EvidenceBundle, plan: AnswerPlan, config: GenerationConfig
    ) -> ContextOptimizationResult:
        """Optimize a bundle's evidence for one generation call.

        Args:
            bundle: The evidence retrieved for this question (Module 9's output).
            plan: The answer plan this context is being built to satisfy.
            config: Generation configuration -- `context_window` and
                `max_tokens` determine how much evidence budget is available.

        Returns:
            The optimized context, ready for Prompt Composition.
        """
        ordered_members = _flatten_in_rank_order(bundle)

        deduplicated, notes = _remove_redundant(ordered_members)

        available_tokens = max(
            config.context_window - config.max_tokens - _RESERVED_TOKENS_FOR_PROMPT_OVERHEAD, 0
        )
        kept, used_tokens = _fit_to_budget(deduplicated, available_tokens)
        if len(kept) < len(deduplicated):
            notes.append(
                f"dropped {len(deduplicated) - len(kept)} evidence item(s): "
                f"context window budget ({available_tokens} estimated tokens) exceeded"
            )

        context_sections = [
            ContextSection(
                citation_label=f"KU{index + 1}",
                knowledge_unit_id=str(member.candidate.knowledge_unit_id),
                text=member.candidate.text,
                modality=member.candidate.modality,
                section_id=(
                    str(member.candidate.section_id) if member.candidate.section_id else None
                ),
            )
            for index, member in enumerate(kept)
        ]
        return ContextOptimizationResult(
            context_sections=context_sections,
            notes=notes,
            estimated_prompt_tokens=used_tokens,
            total_candidates_considered=len(ordered_members),
        )


def _flatten_in_rank_order(bundle: EvidenceBundle) -> list[ScoredCandidate]:
    """Flatten evidence groups into one ordered list: group rank order,
    primary before supporting within each group -- preserving both Module
    9's ranking and its provenance grouping."""
    ordered: list[ScoredCandidate] = []
    for group in bundle.evidence_groups:
        ordered.append(group.primary)
        ordered.extend(group.supporting)
    return ordered


def _remove_redundant(
    members: list[ScoredCandidate],
) -> tuple[list[ScoredCandidate], list[str]]:
    """Drop a member whose text is fully contained in an already-kept
    member's text -- a strict, deterministic overlap relation (exact
    substring containment), never a fuzzy similarity threshold."""
    kept: list[ScoredCandidate] = []
    notes: list[str] = []
    for member in members:
        text = member.candidate.text
        if any(text in existing.candidate.text for existing in kept):
            notes.append(
                f"removed {member.candidate.knowledge_unit_id} as redundant: "
                f"its text is already contained in another kept item"
            )
            continue
        kept.append(member)
    return kept, notes


def _fit_to_budget(
    members: list[ScoredCandidate], available_tokens: int
) -> tuple[list[ScoredCandidate], int]:
    kept: list[ScoredCandidate] = []
    used_tokens = 0
    for member in members:
        estimated = _estimate_tokens(member.candidate.text)
        if used_tokens + estimated > available_tokens:
            continue
        kept.append(member)
        used_tokens += estimated
    return kept, used_tokens


def _estimate_tokens(text: str) -> int:
    return max(len(text) // _CHARS_PER_TOKEN_ESTIMATE, 1)
