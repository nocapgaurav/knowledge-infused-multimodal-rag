"""Phase 4: Prompt Composition.

Never concatenates strings into a prompt. Builds a `PromptContext` --
system prompt, grounding rules, the answer plan, optimized context, and
formatting rules as distinct fields -- and lets each provider render it
into its own wire format (see `providers/ollama_provider.py`).
"""

from backend.generation.models.answer_plan import AnswerPlan, QuestionType
from backend.generation.models.prompt_context import ContextSection, PromptContext

PROMPT_VERSION = "1.0"
"""Version of this module's own prompt template and rule set -- bumped
when the wording or structure of the rules changes, independently of the
`PromptContext`/`GroundedResponse` schema versions."""

_SYSTEM_PROMPT = (
    "You are a scientific research assistant. You answer questions ONLY using "
    "the evidence explicitly provided to you. You never use outside knowledge, "
    "never speculate, and never fill gaps in the evidence with assumptions."
)

_BASE_GROUNDING_RULES = (
    "Cite every factual claim using the evidence labels shown (e.g. [KU1]). "
    "Never invent a citation label that was not shown to you.",
    "If the evidence does not fully answer the question, explicitly state what "
    "is missing rather than guessing or generalizing beyond it.",
    "Never state a fact that is not directly supported by a cited piece of evidence.",
    "Prefer direct quotation or close paraphrase of the evidence over broad generalization.",
)

_NO_EVIDENCE_GROUNDING_RULE = (
    "No evidence was found for this question. State plainly that the available "
    "material does not answer the question -- do not answer from general knowledge."
)

_BASE_FORMATTING_RULES = (
    "Write the detailed answer as clear, well-structured prose.",
    "Use only the evidence labels provided; never write a raw knowledge unit id.",
)

_QUESTION_TYPE_FORMATTING_RULES: dict[QuestionType, tuple[str, ...]] = {
    QuestionType.COMPARATIVE: (
        "Present the comparison in terms of the specific similarities and "
        "differences the evidence supports.",
    ),
    QuestionType.PROCEDURAL: (
        "Present the answer as an ordered sequence of steps if the evidence describes one.",
    ),
    QuestionType.MULTI_HOP: (
        "Explain how the different pieces of cited evidence connect to form the answer.",
    ),
    QuestionType.FIGURE_CENTRIC: (
        "Describe what the cited figure shows before drawing conclusions from it.",
    ),
    QuestionType.TABLE_CENTRIC: (
        "Describe the relevant cited table values before drawing conclusions from them.",
    ),
    QuestionType.CITATION_CENTRIC: (
        "Be explicit about which evidence label supports which specific statement.",
    ),
}


class PromptComposer:
    """Composes a structured PromptContext from a plan and optimized evidence."""

    def compose(
        self, query: str, plan: AnswerPlan, context_sections: list[ContextSection]
    ) -> PromptContext:
        """Compose the structured prompt for one generation call.

        Args:
            query: The user's question, verbatim.
            plan: The answer plan this prompt is built to satisfy.
            context_sections: The optimized evidence (Phase 3's output).

        Returns:
            The complete, structured prompt, ready for Prompt Validation.
        """
        grounding_rules: tuple[str, ...] = _BASE_GROUNDING_RULES
        if not context_sections:
            grounding_rules = (*grounding_rules, _NO_EVIDENCE_GROUNDING_RULE)

        formatting_rules = (
            *_BASE_FORMATTING_RULES,
            *_QUESTION_TYPE_FORMATTING_RULES.get(plan.question_type, ()),
        )

        return PromptContext(
            system_prompt=_SYSTEM_PROMPT,
            grounding_rules=grounding_rules,
            answer_plan=plan,
            context_sections=tuple(context_sections),
            formatting_rules=formatting_rules,
            user_question=query,
        )
