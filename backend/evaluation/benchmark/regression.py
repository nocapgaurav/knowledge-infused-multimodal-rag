"""Compares two benchmark runs (Benchmark A -> Benchmark B) and reports
quality improvements, quality regressions, and latency regressions.

`MetricComparison`/`RegressionReport` live here, not in `models/`, for the
same reason `ExpansionResult`/`AssemblyResult` lived beside Module 9/10's
own phase logic rather than in their `models/` packages: these are the
output of one specific computation (comparing two summaries), not a core
domain object that flows through the rest of the system.
"""

from dataclasses import dataclass
from typing import Literal

from backend.evaluation.models.evaluation_summary import EvaluationSummary

_REGRESSION_THRESHOLD_PERCENT = 5.0
"""A named, documented threshold: a metric change smaller than this is
treated as noise (e.g. minor LLM sampling variation at low temperature),
not a real improvement or regression. Applied uniformly as a relative
percentage rather than classifying which metrics are naturally bounded to
`[0, 1]` versus unbounded (latency, memory) -- simpler, and still
defensible for both kinds of metric."""

_ZERO_BASELINE_ABSOLUTE_THRESHOLD = 0.01
"""Fallback threshold when the baseline value is exactly zero, where a
relative percentage change is undefined (division by zero)."""

_LOWER_IS_BETTER_METRICS = frozenset(
    {
        "unsupported_claim_rate",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "end_to_end_latency_ms",
        "peak_memory_mb",
        "cpu_time_ms",
    }
)
"""Metrics where a decrease is an improvement -- every other metric in
this suite is "higher is better" by construction."""

Direction = Literal["improvement", "regression", "unchanged"]


@dataclass(frozen=True)
class MetricComparison:
    """The comparison of one metric between a baseline and a candidate benchmark run.

    Attributes:
        metric_name: Name of the metric (e.g. "precision@5", "grounding_accuracy").
        baseline_value: The metric's value in the baseline run.
        candidate_value: The metric's value in the candidate run.
        delta: `candidate_value - baseline_value`.
        delta_percent: `delta` as a percentage of `baseline_value`, or of
            `_ZERO_BASELINE_ABSOLUTE_THRESHOLD` if the baseline is zero.
        direction: Whether the candidate is better, worse, or unchanged
            relative to the baseline, accounting for each metric's polarity.
    """

    metric_name: str
    baseline_value: float
    candidate_value: float
    delta: float
    delta_percent: float
    direction: Direction


@dataclass(frozen=True)
class RegressionReport:
    """The complete comparison between two benchmark runs.

    Attributes:
        baseline_benchmark_id: Identifier of the baseline run.
        candidate_benchmark_id: Identifier of the candidate run.
        dataset_versions_match: Whether both runs used the identical
            dataset content -- a regression conclusion is only meaningful
            when this is `True`.
        retrieval_comparisons: Per-metric comparisons for hybrid retrieval.
        generation_comparisons: Per-metric comparisons for generation quality.
        performance_comparisons: Per-metric comparisons for latency/memory/CPU.
    """

    baseline_benchmark_id: str
    candidate_benchmark_id: str
    dataset_versions_match: bool
    retrieval_comparisons: tuple[MetricComparison, ...]
    generation_comparisons: tuple[MetricComparison, ...]
    performance_comparisons: tuple[MetricComparison, ...]

    @property
    def quality_improvements(self) -> tuple[str, ...]:
        """Names of metrics (retrieval + generation) that improved."""
        return tuple(
            comparison.metric_name
            for comparison in (*self.retrieval_comparisons, *self.generation_comparisons)
            if comparison.direction == "improvement"
        )

    @property
    def quality_regressions(self) -> tuple[str, ...]:
        """Names of metrics (retrieval + generation) that regressed."""
        return tuple(
            comparison.metric_name
            for comparison in (*self.retrieval_comparisons, *self.generation_comparisons)
            if comparison.direction == "regression"
        )

    @property
    def latency_regressions(self) -> tuple[str, ...]:
        """Names of performance metrics that regressed."""
        return tuple(
            comparison.metric_name
            for comparison in self.performance_comparisons
            if comparison.direction == "regression"
        )


def compare(baseline: EvaluationSummary, candidate: EvaluationSummary) -> RegressionReport:
    """Compare two benchmark runs.

    Args:
        baseline: The earlier (reference) benchmark run.
        candidate: The later (candidate) benchmark run being evaluated
            against the baseline.

    Returns:
        The complete regression report.
    """
    return RegressionReport(
        baseline_benchmark_id=baseline.manifest.benchmark_id,
        candidate_benchmark_id=candidate.manifest.benchmark_id,
        dataset_versions_match=(
            baseline.manifest.dataset_version == candidate.manifest.dataset_version
        ),
        retrieval_comparisons=_compare_aggregate(
            baseline.hybrid_retrieval_aggregate, candidate.hybrid_retrieval_aggregate
        ),
        generation_comparisons=_compare_aggregate(
            baseline.generation_aggregate, candidate.generation_aggregate
        ),
        performance_comparisons=_compare_aggregate(
            baseline.performance_aggregate, candidate.performance_aggregate
        ),
    )


def _compare_aggregate(
    baseline_aggregate: dict[str, float], candidate_aggregate: dict[str, float]
) -> tuple[MetricComparison, ...]:
    shared_metric_names = sorted(set(baseline_aggregate) & set(candidate_aggregate))
    return tuple(
        _compare_metric(name, baseline_aggregate[name], candidate_aggregate[name])
        for name in shared_metric_names
    )


def _compare_metric(name: str, baseline_value: float, candidate_value: float) -> MetricComparison:
    delta = candidate_value - baseline_value
    if baseline_value == 0.0:
        delta_percent = 0.0 if delta == 0.0 else (delta / _ZERO_BASELINE_ABSOLUTE_THRESHOLD) * 100
    else:
        delta_percent = (delta / abs(baseline_value)) * 100

    direction = _classify_direction(name, delta_percent)
    return MetricComparison(
        metric_name=name,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        delta=delta,
        delta_percent=delta_percent,
        direction=direction,
    )


def _classify_direction(metric_name: str, delta_percent: float) -> Direction:
    if abs(delta_percent) < _REGRESSION_THRESHOLD_PERCENT:
        return "unchanged"
    improved = delta_percent < 0 if metric_name in _LOWER_IS_BETTER_METRICS else delta_percent > 0
    return "improvement" if improved else "regression"
