"""EvaluationSummary: the complete record of one benchmark run.

The single artifact persisted as `benchmark.json` and rendered into
`benchmark.md`/`benchmark.html` -- every report format is a derived view
of exactly this object, never independently computed.
"""

from pydantic import BaseModel, ConfigDict, Field

from backend.evaluation.models.benchmark_manifest import BenchmarkManifest
from backend.evaluation.models.benchmark_result import BenchmarkResult


class CaseFailure(BaseModel):
    """A case that could not be completed through the real pipeline.

    Kept separate from `BenchmarkResult` rather than making that model's
    fields nullable: a single case failing for an infrastructure reason
    (e.g. a transient provider error) should not force every consumer of
    a successful result to handle absent metrics -- aggregates are
    computed only over cases that actually completed.

    Attributes:
        case_id: Identifier of the case that failed.
        reason: Human-readable failure reason.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class EvaluationSummary(BaseModel):
    """The complete outcome of one benchmark run, cases and aggregates together.

    Attributes:
        manifest: Versioning and provenance for this run.
        case_results: Every successfully completed case's result, in dataset order.
        failed_cases: Cases that could not be completed, with why. Excluded
            from every aggregate and breakdown below.
        dense_retrieval_aggregate: Metric name -> mean value, across all
            cases, for dense-only retrieval.
        hybrid_retrieval_aggregate: Metric name -> mean value, across all
            cases, for hybrid retrieval.
        generation_aggregate: Metric name -> mean value, across all cases,
            for generation quality.
        performance_aggregate: Metric name -> mean value, across all
            cases, for latency/memory/CPU. Also includes `throughput`,
            computed once across the whole run rather than per case.
        answer_status_accuracy: Fraction of cases whose produced
            `AnswerStatus` matched the case's expected status exactly.
        breakdown_by_difficulty: Difficulty -> (metric name -> mean value),
            across the generation and hybrid retrieval metrics, for
            segmented reporting.
        breakdown_by_category: Category -> (metric name -> mean value),
            across the generation and hybrid retrieval metrics, for
            segmented reporting.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: BenchmarkManifest
    case_results: tuple[BenchmarkResult, ...]
    failed_cases: tuple[CaseFailure, ...] = Field(default_factory=tuple)
    dense_retrieval_aggregate: dict[str, float]
    hybrid_retrieval_aggregate: dict[str, float]
    generation_aggregate: dict[str, float]
    performance_aggregate: dict[str, float]
    answer_status_accuracy: float = Field(ge=0.0, le=1.0)
    breakdown_by_difficulty: dict[str, dict[str, float]]
    breakdown_by_category: dict[str, dict[str, float]]
