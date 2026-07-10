"""EvaluationCase: one labeled question in an evaluation dataset."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import PaperId
from backend.generation.models import AnswerStatus


class Difficulty(StrEnum):
    """How difficult a case is expected to be for the pipeline.

    Used only for segmented reporting (breaking metrics down by
    difficulty) -- it never affects how a case is evaluated.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvaluationCase(BaseModel):
    """One labeled question, with the ground truth needed to score the
    pipeline's real output against it.

    Attributes:
        case_id: Stable, human-assigned identifier (e.g. "case-001") --
            deliberately not derived from the question text, so it
            survives minor wording edits across dataset versions.
        question: The question to ask.
        document_id: Identifier of the document to ask it against.
        ground_truth_answer: A reference answer, for qualitative
            side-by-side display in reports -- never used as a numeric
            metric input, to avoid reintroducing a fuzzy text-similarity
            score in place of the deterministic, knowledge-unit-based
            metrics this module uses instead.
        expected_knowledge_units: Knowledge unit ids relevant to this
            question -- the ground truth retrieval metrics (Precision@K,
            Recall@K, MRR, NDCG, Hit Rate) are computed against.
        expected_citations: Knowledge unit ids the ideal answer should
            cite -- the ground truth generation's Evidence Coverage and
            Answer Completeness metrics are computed against. Distinct
            from `expected_knowledge_units`: retrieval may reasonably
            surface more relevant material than an answer needs to cite.
        expected_answer_status: The `AnswerStatus` a correctly-behaving
            pipeline should produce for this question.
        difficulty: Expected difficulty, for segmented reporting.
        category: Free-form category label (e.g. "factual",
            "comparative"), for segmented reporting. Not a closed enum --
            dataset authors define their own categories.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    document_id: PaperId
    ground_truth_answer: str = Field(min_length=1)
    expected_knowledge_units: tuple[str, ...] = Field(min_length=1)
    expected_citations: tuple[str, ...] = Field(min_length=1)
    expected_answer_status: AnswerStatus
    difficulty: Difficulty
    category: str = Field(min_length=1)
