"""AnswerPlan: the deterministic decision of how a question should be answered,
made before any prompt exists."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import ChunkModality


class QuestionType(StrEnum):
    """The kind of question being asked -- describes the input.

    Deliberately closed and small: each member must be detectable from
    the question text and the shape of the retrieved evidence alone, with
    no LLM involved in the classification itself.
    """

    FACTUAL = "factual"
    DESCRIPTIVE = "descriptive"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    MULTI_HOP = "multi_hop"
    FIGURE_CENTRIC = "figure_centric"
    TABLE_CENTRIC = "table_centric"
    CITATION_CENTRIC = "citation_centric"


class ExpectedAnswerType(StrEnum):
    """The shape the final answer should take -- describes the output.

    Kept distinct from `QuestionType` even though today's mapping is 1:1:
    the input classification and the output shape are different concerns,
    and a future refinement (e.g. two question types converging on one
    answer shape) should not require collapsing them into one enum.
    """

    SHORT_FACTUAL = "short_factual"
    NARRATIVE_DESCRIPTION = "narrative_description"
    STRUCTURED_COMPARISON = "structured_comparison"
    STEP_BY_STEP = "step_by_step"
    MULTI_PART_SYNTHESIS = "multi_part_synthesis"
    FIGURE_EXPLANATION = "figure_explanation"
    TABLE_EXPLANATION = "table_explanation"
    CITATION_SUMMARY = "citation_summary"


class AnswerSection(StrEnum):
    """A named section the final formatted response may include."""

    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANSWER = "detailed_answer"
    SUPPORTING_EVIDENCE = "supporting_evidence"
    LIMITATIONS = "limitations"
    REFERENCES = "references"
    WARNINGS = "warnings"


class AnswerPlan(BaseModel):
    """The deterministic plan for how a question should be answered.

    Produced once, before Context Optimization or Prompt Composition run
    -- both depend on knowing what the answer needs to accomplish.

    Attributes:
        question_type: The kind of question being asked.
        expected_answer_type: The shape the final answer should take.
        required_evidence_groups: Minimum number of evidence groups needed
            to answer this question type adequately. Used by Answer
            Quality Assessment to judge evidence completeness.
        required_modalities: Content modalities the question calls for
            (e.g. `FIGURE_CENTRIC` requires `ChunkModality.FIGURE`).
            Empty means no specific modality is required.
        requires_citations: Whether every claim in the answer must carry
            an inline citation. `False` only for questions whose answer is
            trivially derivable from a single, obviously-cited source.
        expected_sections: Which structural sections the final response
            should include, in order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_type: QuestionType
    expected_answer_type: ExpectedAnswerType
    required_evidence_groups: int = Field(ge=1)
    required_modalities: tuple[ChunkModality, ...] = Field(default_factory=tuple)
    requires_citations: bool = True
    expected_sections: tuple[AnswerSection, ...] = Field(min_length=1)
