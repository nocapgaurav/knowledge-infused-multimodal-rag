"""BenchmarkResult: the complete, per-case outcome of one evaluation run."""

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import PaperId
from backend.evaluation.models.evaluation_case import Difficulty
from backend.generation.models import AnswerStatus


class RetrievalCaseMetrics(BaseModel):
    """Rank-based retrieval metrics for one case, at each evaluated cutoff.

    Attributes:
        precision_at_k: Cutoff `k` -> Precision@k.
        recall_at_k: Cutoff `k` -> Recall@k.
        reciprocal_rank: This case's reciprocal rank (averaged across
            cases at summary time to produce MRR).
        ndcg_at_k: Cutoff `k` -> NDCG@k.
        hit_rate_at_k: Cutoff `k` -> Hit Rate@k.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]


class GenerationCaseMetrics(BaseModel):
    """Generation quality metrics for one case.

    Attributes:
        grounding_accuracy: Fraction of claims that passed grounding validation.
        citation_accuracy: Fraction of citation labels used that resolved to real evidence.
        evidence_coverage: Of what was cited, the fraction that was expected (precision-flavored).
        answer_completeness: Of what was expected, the fraction that was cited (recall-flavored).
        unsupported_claim_rate: Fraction of claims that failed grounding, for any reason.
        answer_status_correct: Whether the produced `AnswerStatus` matched
            `EvaluationCase.expected_answer_status` exactly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    grounding_accuracy: float = Field(ge=0.0, le=1.0)
    citation_accuracy: float = Field(ge=0.0, le=1.0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    answer_completeness: float = Field(ge=0.0, le=1.0)
    unsupported_claim_rate: float = Field(ge=0.0, le=1.0)
    answer_status_correct: bool


class PerformanceCaseMetrics(BaseModel):
    """System performance metrics for one case.

    Attributes:
        retrieval_latency_ms: Wall-clock time for the retrieval call.
        generation_latency_ms: Wall-clock time for the generation call.
        end_to_end_latency_ms: Total wall-clock time for this case.
        peak_memory_mb: Process peak RSS at the time this case completed
            -- a watermark since process start, not an isolated
            per-case delta (see the final report's known limitations).
        cpu_time_ms: Process user+system CPU time consumed during this case.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    retrieval_latency_ms: float = Field(ge=0.0)
    generation_latency_ms: float = Field(ge=0.0)
    end_to_end_latency_ms: float = Field(ge=0.0)
    peak_memory_mb: float = Field(ge=0.0)
    cpu_time_ms: float = Field(ge=0.0)


class BenchmarkResult(BaseModel):
    """The complete outcome of running one evaluation case through the real pipeline.

    Attributes:
        case_id: Identifier of the evaluated case.
        question: The question asked.
        document_id: Identifier of the document it was asked against.
        difficulty: The case's expected difficulty, carried through for segmented reporting.
        category: The case's category, carried through for segmented reporting.
        dense_retrieval_metrics: Retrieval metrics using dense-only retrieval
            (`ExpansionBudget(max_depth=0)`), for comparison against hybrid.
        hybrid_retrieval_metrics: Retrieval metrics using the full hybrid
            pipeline -- the strategy generation actually runs on.
        generation_metrics: Generation quality metrics for the hybrid-retrieved answer.
        performance_metrics: Latency, memory, and CPU for this case.
        generated_answer: The real, unedited generated answer text.
        ground_truth_answer: The case's reference answer, for side-by-side display.
        answer_status: The `AnswerStatus` the pipeline actually produced.
        expected_answer_status: The `AnswerStatus` the case expected.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    document_id: PaperId
    difficulty: Difficulty
    category: str = Field(min_length=1)
    dense_retrieval_metrics: RetrievalCaseMetrics
    hybrid_retrieval_metrics: RetrievalCaseMetrics
    generation_metrics: GenerationCaseMetrics
    performance_metrics: PerformanceCaseMetrics
    generated_answer: str
    ground_truth_answer: str
    answer_status: AnswerStatus
    expected_answer_status: AnswerStatus
