"""Runs an evaluation dataset through the real, unmodified retrieval and
generation pipelines and computes every metric for each case.

Never calls anything but the same public services a real caller would:
`RetrievalService.retrieve` and `GenerationService.generate`. "Dense-only"
retrieval for comparison is achieved by injecting a second
`RetrievalService` instance constructed with a zero-depth expansion
budget (confirmed by real testing: `ExpansionBudget(max_depth=0)` makes
Module 9's own expansion loop discover nothing new) -- not a special code
path, just a different, equally real configuration of the same class.

A single case failing (a transient provider error, an unindexed document)
does not abort the run -- it is recorded as a `CaseFailure` and excluded
from every aggregate, matching the bounded-fault-tolerance pattern used
throughout this project's own batch-processing code.
"""

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import mean

from backend.evaluation.metrics.answer_completeness import answer_completeness
from backend.evaluation.metrics.citation_accuracy import citation_accuracy
from backend.evaluation.metrics.cpu import cpu_time_delta_ms, snapshot_cpu_time
from backend.evaluation.metrics.evidence_coverage import evidence_coverage
from backend.evaluation.metrics.grounding_accuracy import grounding_accuracy
from backend.evaluation.metrics.hit_rate import hit_rate_at_k
from backend.evaluation.metrics.memory import current_peak_memory_mb
from backend.evaluation.metrics.mrr import reciprocal_rank
from backend.evaluation.metrics.ndcg import ndcg_at_k
from backend.evaluation.metrics.precision import precision_at_k
from backend.evaluation.metrics.recall import recall_at_k
from backend.evaluation.metrics.unsupported_claim_rate import unsupported_claim_rate
from backend.evaluation.models.benchmark_result import (
    BenchmarkResult,
    GenerationCaseMetrics,
    PerformanceCaseMetrics,
    RetrievalCaseMetrics,
)
from backend.evaluation.models.evaluation_case import EvaluationCase
from backend.evaluation.models.evaluation_summary import CaseFailure
from backend.generation.exceptions import GenerationError
from backend.generation.models import AnswerStatus, GenerationConfig, GroundedResponse
from backend.generation.services.generation_service import GenerationService
from backend.retrieval.exceptions import RetrievalError
from backend.retrieval.models import EvidenceBundle
from backend.retrieval.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

DEFAULT_K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True)
class BenchmarkRunResult:
    """The outcome of running a full evaluation dataset once.

    Attributes:
        case_results: Every successfully completed case's result.
        failed_cases: Cases that could not be completed, with why.
        dense_retrieval_aggregate: Metric name -> mean, dense-only retrieval.
        hybrid_retrieval_aggregate: Metric name -> mean, hybrid retrieval.
        generation_aggregate: Metric name -> mean, generation quality.
        performance_aggregate: Metric name -> aggregate, latency/memory/CPU.
        answer_status_accuracy: Fraction of cases whose produced status matched expectation.
        breakdown_by_difficulty: Difficulty -> (metric name -> mean).
        breakdown_by_category: Category -> (metric name -> mean).
        total_duration_seconds: Total wall-clock time for the whole run.
        retrieval_strategy_version: The hybrid retrieval strategy version
            observed during this run, or `None` if every case failed.
        generation_prompt_version: The generation prompt version observed
            during this run, or `None` if every case failed.
    """

    case_results: list[BenchmarkResult]
    failed_cases: list[CaseFailure]
    dense_retrieval_aggregate: dict[str, float]
    hybrid_retrieval_aggregate: dict[str, float]
    generation_aggregate: dict[str, float]
    performance_aggregate: dict[str, float]
    answer_status_accuracy: float
    breakdown_by_difficulty: dict[str, dict[str, float]]
    breakdown_by_category: dict[str, dict[str, float]]
    total_duration_seconds: float
    retrieval_strategy_version: str | None
    generation_prompt_version: str | None


class BenchmarkRunner:
    """Runs an evaluation dataset through the real pipeline and computes metrics."""

    def __init__(
        self,
        dense_retrieval_service: RetrievalService,
        hybrid_retrieval_service: RetrievalService,
        generation_service: GenerationService,
        generation_config: GenerationConfig,
        k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    ) -> None:
        """Initialize the runner.

        Args:
            dense_retrieval_service: A `RetrievalService` configured with
                a zero-depth expansion budget, for dense-only comparison.
            hybrid_retrieval_service: A `RetrievalService` configured
                normally -- the strategy generation actually runs on.
            generation_service: The real generation service.
            generation_config: Generation parameters to use for every case.
            k_values: Cutoffs to compute rank-based retrieval metrics at.
        """
        self._dense_retrieval_service = dense_retrieval_service
        self._hybrid_retrieval_service = hybrid_retrieval_service
        self._generation_service = generation_service
        self._generation_config = generation_config
        self._k_values = k_values

    def run(self, cases: Sequence[EvaluationCase]) -> BenchmarkRunResult:
        """Run every case in a dataset through the real pipeline.

        Args:
            cases: The evaluation cases to run.

        Returns:
            The complete outcome of the run, aggregated and broken down.
        """
        started_at = time.perf_counter()
        results: list[BenchmarkResult] = []
        failures: list[CaseFailure] = []
        retrieval_strategy_version: str | None = None
        generation_prompt_version: str | None = None

        for case in cases:
            try:
                outcome = self._run_case(case)
            except (RetrievalError, GenerationError) as exc:
                logger.error(
                    "evaluation case failed", exc_info=True, extra={"case_id": case.case_id}
                )
                failures.append(CaseFailure(case_id=case.case_id, reason=str(exc)))
                continue
            results.append(outcome.result)
            if retrieval_strategy_version is None:
                retrieval_strategy_version = outcome.retrieval_strategy_version
            if generation_prompt_version is None:
                generation_prompt_version = outcome.generation_prompt_version

        total_duration_seconds = time.perf_counter() - started_at

        return BenchmarkRunResult(
            case_results=results,
            failed_cases=failures,
            dense_retrieval_aggregate=_aggregate_retrieval(
                results, lambda r: r.dense_retrieval_metrics, self._k_values
            ),
            hybrid_retrieval_aggregate=_aggregate_retrieval(
                results, lambda r: r.hybrid_retrieval_metrics, self._k_values
            ),
            generation_aggregate=_aggregate_generation(results),
            performance_aggregate=_aggregate_performance(results),
            answer_status_accuracy=_answer_status_accuracy(results),
            breakdown_by_difficulty=_breakdown_by(results, lambda r: r.difficulty.value),
            breakdown_by_category=_breakdown_by(results, lambda r: r.category),
            total_duration_seconds=total_duration_seconds,
            retrieval_strategy_version=retrieval_strategy_version,
            generation_prompt_version=generation_prompt_version,
        )

    def _run_case(self, case: EvaluationCase) -> "_CaseRunOutcome":
        cpu_before = snapshot_cpu_time()
        case_started = time.perf_counter()

        dense_bundle = self._dense_retrieval_service.retrieve(case.document_id, case.question)

        hybrid_started = time.perf_counter()
        hybrid_bundle = self._hybrid_retrieval_service.retrieve(case.document_id, case.question)
        retrieval_latency_ms = (time.perf_counter() - hybrid_started) * 1000

        generation_started = time.perf_counter()
        response = self._generation_service.generate(hybrid_bundle, self._generation_config)
        generation_latency_ms = (time.perf_counter() - generation_started) * 1000

        end_to_end_latency_ms = (time.perf_counter() - case_started) * 1000
        cpu_after = snapshot_cpu_time()

        expected_units = set(case.expected_knowledge_units)
        expected_citations = set(case.expected_citations)

        result = BenchmarkResult(
            case_id=case.case_id,
            question=case.question,
            document_id=case.document_id,
            difficulty=case.difficulty,
            category=case.category,
            dense_retrieval_metrics=self._compute_retrieval_metrics(dense_bundle, expected_units),
            hybrid_retrieval_metrics=self._compute_retrieval_metrics(hybrid_bundle, expected_units),
            generation_metrics=self._compute_generation_metrics(
                response, expected_citations, case.expected_answer_status
            ),
            performance_metrics=PerformanceCaseMetrics(
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                end_to_end_latency_ms=end_to_end_latency_ms,
                peak_memory_mb=current_peak_memory_mb(),
                cpu_time_ms=cpu_time_delta_ms(cpu_before, cpu_after),
            ),
            generated_answer=response.answer,
            ground_truth_answer=case.ground_truth_answer,
            answer_status=response.answer_status,
            expected_answer_status=case.expected_answer_status,
        )
        return _CaseRunOutcome(
            result=result,
            retrieval_strategy_version=hybrid_bundle.manifest.retrieval_strategy_version,
            generation_prompt_version=response.prompt_version,
        )

    def _compute_retrieval_metrics(
        self, bundle: EvidenceBundle, expected_units: set[str]
    ) -> RetrievalCaseMetrics:
        retrieved_ids = _ranked_knowledge_unit_ids(bundle)
        return RetrievalCaseMetrics(
            precision_at_k={
                k: precision_at_k(retrieved_ids, expected_units, k) for k in self._k_values
            },
            recall_at_k={k: recall_at_k(retrieved_ids, expected_units, k) for k in self._k_values},
            reciprocal_rank=reciprocal_rank(retrieved_ids, expected_units),
            ndcg_at_k={k: ndcg_at_k(retrieved_ids, expected_units, k) for k in self._k_values},
            hit_rate_at_k={
                k: hit_rate_at_k(retrieved_ids, expected_units, k) for k in self._k_values
            },
        )

    def _compute_generation_metrics(
        self,
        response: GroundedResponse,
        expected_citations: set[str],
        expected_status: AnswerStatus,
    ) -> GenerationCaseMetrics:
        stats = response.generation_statistics
        cited_ids = {citation.knowledge_unit_id for citation in response.resolved_citations}
        return GenerationCaseMetrics(
            grounding_accuracy=grounding_accuracy(stats.claims_grounded, stats.claims_total),
            citation_accuracy=citation_accuracy(
                stats.citations_resolved, stats.citations_unresolved
            ),
            evidence_coverage=evidence_coverage(cited_ids, expected_citations),
            answer_completeness=answer_completeness(cited_ids, expected_citations),
            unsupported_claim_rate=unsupported_claim_rate(
                stats.claims_grounded, stats.claims_total
            ),
            answer_status_correct=response.answer_status == expected_status,
        )


@dataclass(frozen=True)
class _CaseRunOutcome:
    """One case's result, plus the pipeline versions observed producing
    it -- captured here because `BenchmarkResult` itself doesn't carry
    them; a benchmark run records only the first successful case's
    versions, since one run targets one deployed pipeline in practice."""

    result: BenchmarkResult
    retrieval_strategy_version: str
    generation_prompt_version: str


def _ranked_knowledge_unit_ids(bundle: EvidenceBundle) -> list[str]:
    """Flatten evidence groups into one ordered list: group rank order,
    primary before supporting within each group -- the same "flatten in
    rank order" reading Module 10's own Context Optimization uses."""
    ordered: list[str] = []
    for group in bundle.evidence_groups:
        ordered.append(str(group.primary.candidate.knowledge_unit_id))
        ordered.extend(str(member.candidate.knowledge_unit_id) for member in group.supporting)
    return ordered


def _aggregate_retrieval(
    results: list[BenchmarkResult],
    select: Callable[[BenchmarkResult], RetrievalCaseMetrics],
    k_values: tuple[int, ...],
) -> dict[str, float]:
    if not results:
        return {}
    metrics = [select(result) for result in results]
    aggregate = {"mrr": mean(m.reciprocal_rank for m in metrics)}
    for k in k_values:
        aggregate[f"precision@{k}"] = mean(m.precision_at_k[k] for m in metrics)
        aggregate[f"recall@{k}"] = mean(m.recall_at_k[k] for m in metrics)
        aggregate[f"ndcg@{k}"] = mean(m.ndcg_at_k[k] for m in metrics)
        aggregate[f"hit_rate@{k}"] = mean(m.hit_rate_at_k[k] for m in metrics)
    return aggregate


def _aggregate_generation(results: list[BenchmarkResult]) -> dict[str, float]:
    if not results:
        return {}
    metrics = [result.generation_metrics for result in results]
    return {
        "grounding_accuracy": mean(m.grounding_accuracy for m in metrics),
        "citation_accuracy": mean(m.citation_accuracy for m in metrics),
        "evidence_coverage": mean(m.evidence_coverage for m in metrics),
        "answer_completeness": mean(m.answer_completeness for m in metrics),
        "unsupported_claim_rate": mean(m.unsupported_claim_rate for m in metrics),
    }


def _aggregate_performance(results: list[BenchmarkResult]) -> dict[str, float]:
    if not results:
        return {}
    metrics = [result.performance_metrics for result in results]
    return {
        "retrieval_latency_ms": mean(m.retrieval_latency_ms for m in metrics),
        "generation_latency_ms": mean(m.generation_latency_ms for m in metrics),
        "end_to_end_latency_ms": mean(m.end_to_end_latency_ms for m in metrics),
        # peak_memory_mb is already a watermark (see metrics/memory.py) --
        # the most informative aggregate over a run is the highest
        # watermark observed, not the mean of watermarks.
        "peak_memory_mb": max(m.peak_memory_mb for m in metrics),
        "cpu_time_ms": mean(m.cpu_time_ms for m in metrics),
    }


def _answer_status_accuracy(results: list[BenchmarkResult]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for result in results if result.generation_metrics.answer_status_correct)
    return correct / len(results)


def _breakdown_by(
    results: list[BenchmarkResult], key_fn: Callable[[BenchmarkResult], str]
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        groups[key_fn(result)].append(result)
    return {key: _aggregate_generation(group) for key, group in groups.items()}
