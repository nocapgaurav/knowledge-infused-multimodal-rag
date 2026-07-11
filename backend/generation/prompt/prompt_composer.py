"""Phase 4: Prompt Composition.

Never concatenates strings into a prompt. Builds a `PromptContext` --
system prompt, grounding rules, the answer plan, optimized context, and
formatting rules as distinct fields -- and lets each provider render it
into its own wire format (see `providers/ollama_provider.py`).
"""

from backend.generation.models.answer_plan import AnswerPlan, QuestionType
from backend.generation.models.prompt_context import ContextSection, PromptContext

PROMPT_VERSION = "1.1"
"""Version of this module's own prompt template and rule set -- bumped
when the wording or structure of the rules changes, independently of the
`PromptContext`/`GroundedResponse` schema versions."""

_SYSTEM_PROMPT = (
    "You are an expert researcher explaining a scientific paper to a colleague. "
    "You answer questions ONLY using the evidence explicitly provided to you. "
    "You never use outside knowledge, never speculate, and never fill gaps in "
    "the evidence with assumptions. Within those bounds, you SYNTHESIZE: "
    "connect the cited pieces of evidence to each other, explain what each "
    "cited fact means for the paper as a whole, and say why it matters -- an "
    "educational explanation built entirely from the evidence, not a list of "
    "quotations. Each evidence item is prefixed with its identity in "
    "parentheses, e.g. (Figure 2) or (Section: III. Methodology); use those "
    "identities to relate evidence to the paper's structure."
)

_BASE_GROUNDING_RULES = (
    "Cite every sentence that states a fact, using ONLY the bracketed evidence "
    "labels shown (e.g. [KU1]) -- the parenthesized identity next to each label "
    "is context, not a citation form. A sentence that interprets or connects "
    "already-cited facts should cite the evidence those facts came from. Never "
    "invent a citation label that was not shown to you.",
    "If the evidence does not fully answer the question, explicitly state what "
    "is missing rather than guessing or generalizing beyond it.",
    "Never state a fact that is not directly supported by a cited piece of evidence.",
    "Explain and connect the evidence rather than merely quoting it -- but every "
    "explanation must be traceable to the cited text, never to outside knowledge.",
)

_NO_EVIDENCE_GROUNDING_RULE = (
    "No evidence was found for this question. State plainly that the available "
    "material does not answer the question -- do not answer from general knowledge."
)

_BASE_FORMATTING_RULES = (
    "Write the detailed answer as clear, well-structured prose.",
    "Use only the evidence labels provided; never write a raw knowledge unit id.",
    "When the answer is a list (keywords, authors, steps), cite the evidence "
    "label on the sentence that introduces the list.",
    "Organize naturally for the question: typically a direct answer first, then "
    "the deeper explanation, then how it connects to the rest of the paper, then "
    "any limitations THE EVIDENCE ITSELF states or implies. Skip any part the "
    "evidence cannot support -- never pad with generic statements.",
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
        "Explain the figure like a researcher walking a colleague through it: "
        "what it depicts (from its caption), what purpose it serves in the "
        "paper, and how the surrounding cited text explains its components, "
        "workflow, or significance. Draw ONLY on the cited evidence -- if the "
        "evidence describes the pipeline the figure illustrates, use it; if "
        "something about the figure is not in the evidence, say so plainly.",
    ),
    QuestionType.TABLE_CENTRIC: (
        "Interpret the table like a researcher: what it compares or reports, "
        "the concrete values, trends, or differences visible in the cited "
        "table content, why the paper includes it, and what the surrounding "
        "cited discussion concludes from it. State trends only when the cited "
        "rows actually show them.",
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
