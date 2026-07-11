"""Phase 2: Answer Planning.

Classifies the question and decides what the answer needs -- before any
prompt exists. Classification is rule-based (keyword markers plus the
bundle's own already-computed evidence characteristics), never an LLM
call: this decision must be as deterministic and reproducible as
retrieval itself.
"""

from backend.domain import ChunkModality
from backend.generation.models.answer_plan import (
    AnswerPlan,
    AnswerSection,
    ExpectedAnswerType,
    QuestionType,
)
from backend.retrieval.models import DiscoveryMethod, EvidenceBundle

_COMPARATIVE_MARKERS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "vs.",
    "difference between",
    "differences between",
    "better than",
    "compared to",
    "compared with",
)
_PROCEDURAL_MARKERS = (
    "how to",
    "how do",
    "how does one",
    "steps to",
    "procedure for",
    "process for",
)
_CITATION_MARKERS = ("who wrote", "which paper", "cite", "citation", "reference", "according to")
_FIGURE_MARKERS = (
    "figure",
    "chart",
    "graph",
    "diagram",
    "image",
    "plot",
    "architecture",
    "workflow",
    "illustration",
)
_TABLE_MARKERS = ("table",)
_DESCRIPTIVE_MARKERS = ("what is", "what are", "describe", "explain", "overview of")

_MULTI_HOP_MINIMUM_EVIDENCE_GROUPS = 3

_EXPECTED_ANSWER_TYPE: dict[QuestionType, ExpectedAnswerType] = {
    QuestionType.FACTUAL: ExpectedAnswerType.SHORT_FACTUAL,
    QuestionType.DESCRIPTIVE: ExpectedAnswerType.NARRATIVE_DESCRIPTION,
    QuestionType.COMPARATIVE: ExpectedAnswerType.STRUCTURED_COMPARISON,
    QuestionType.PROCEDURAL: ExpectedAnswerType.STEP_BY_STEP,
    QuestionType.MULTI_HOP: ExpectedAnswerType.MULTI_PART_SYNTHESIS,
    QuestionType.FIGURE_CENTRIC: ExpectedAnswerType.FIGURE_EXPLANATION,
    QuestionType.TABLE_CENTRIC: ExpectedAnswerType.TABLE_EXPLANATION,
    QuestionType.CITATION_CENTRIC: ExpectedAnswerType.CITATION_SUMMARY,
}

_REQUIRED_EVIDENCE_GROUPS: dict[QuestionType, int] = {
    QuestionType.FACTUAL: 1,
    QuestionType.DESCRIPTIVE: 1,
    QuestionType.COMPARATIVE: 2,
    QuestionType.PROCEDURAL: 1,
    QuestionType.MULTI_HOP: 2,
    QuestionType.FIGURE_CENTRIC: 1,
    QuestionType.TABLE_CENTRIC: 1,
    QuestionType.CITATION_CENTRIC: 1,
}

_REQUIRED_MODALITIES: dict[QuestionType, tuple[ChunkModality, ...]] = {
    QuestionType.FIGURE_CENTRIC: (ChunkModality.FIGURE,),
    QuestionType.TABLE_CENTRIC: (ChunkModality.TABLE,),
}

_ALL_SECTIONS: tuple[AnswerSection, ...] = tuple(AnswerSection)
"""Every plan expects the full six-section structured response -- a
deliberate, question-type-independent decision (a production scientific
assistant should look the same shape every time), not a limitation of the
planner. `Limitations`/`Warnings` may end up empty, but the section
always exists in the contract Response Formatting must fulfill."""


class AnswerPlanner:
    """Determines how a question should be answered, before any prompt exists."""

    def plan(self, query: str, bundle: EvidenceBundle) -> AnswerPlan:
        """Produce the deterministic answer plan for a question.

        Args:
            query: The user's question, verbatim.
            bundle: The evidence retrieved for this question (Module 9's output).

        Returns:
            The plan Context Optimization and Prompt Composition build toward.
        """
        question_type = _classify_question(query, bundle)
        return AnswerPlan(
            question_type=question_type,
            expected_answer_type=_EXPECTED_ANSWER_TYPE[question_type],
            required_evidence_groups=_REQUIRED_EVIDENCE_GROUPS[question_type],
            required_modalities=_REQUIRED_MODALITIES.get(question_type, ()),
            requires_citations=True,
            expected_sections=_ALL_SECTIONS,
        )


def _classify_question(query: str, bundle: EvidenceBundle) -> QuestionType:
    normalized = f" {query.lower()} "

    if _matches_any(normalized, _COMPARATIVE_MARKERS):
        return QuestionType.COMPARATIVE
    if _matches_any(normalized, _PROCEDURAL_MARKERS):
        return QuestionType.PROCEDURAL
    if _matches_any(normalized, _CITATION_MARKERS):
        return QuestionType.CITATION_CENTRIC
    if _matches_any(normalized, _FIGURE_MARKERS) or _top_modality(bundle) is ChunkModality.FIGURE:
        return QuestionType.FIGURE_CENTRIC
    if _matches_any(normalized, _TABLE_MARKERS) or _top_modality(bundle) is ChunkModality.TABLE:
        return QuestionType.TABLE_CENTRIC
    if _is_multi_hop(bundle):
        return QuestionType.MULTI_HOP
    if _matches_any(normalized, _DESCRIPTIVE_MARKERS):
        return QuestionType.DESCRIPTIVE
    return QuestionType.FACTUAL


def _matches_any(normalized_query: str, markers: tuple[str, ...]) -> bool:
    return any(marker in normalized_query for marker in markers)


def _top_modality(bundle: EvidenceBundle) -> ChunkModality | None:
    if not bundle.evidence_groups:
        return None
    return bundle.evidence_groups[0].primary.candidate.modality


def _is_multi_hop(bundle: EvidenceBundle) -> bool:
    """Whether answering requires synthesizing several independently-anchored
    pieces of evidence, at least one of which was only found by following a
    graph relationship rather than by directly matching the question.
    """
    if len(bundle.evidence_groups) < _MULTI_HOP_MINIMUM_EVIDENCE_GROUPS:
        return False
    return any(
        candidate.discovery_method is DiscoveryMethod.GRAPH_EXPANSION
        for candidate in bundle.candidates
    )
