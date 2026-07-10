"""Unit tests for Benchmark A -> Benchmark B regression comparison."""

from backend.evaluation.benchmark.regression import compare

from ._helpers import build_summary


def test_compare_flags_higher_is_better_metric_improvement() -> None:
    baseline = build_summary(generation_aggregate={"grounding_accuracy": 0.70})
    candidate = build_summary(generation_aggregate={"grounding_accuracy": 0.90})

    report = compare(baseline, candidate)

    assert "grounding_accuracy" in report.quality_improvements
    assert "grounding_accuracy" not in report.quality_regressions


def test_compare_flags_higher_is_better_metric_regression() -> None:
    baseline = build_summary(generation_aggregate={"grounding_accuracy": 0.90})
    candidate = build_summary(generation_aggregate={"grounding_accuracy": 0.70})

    report = compare(baseline, candidate)

    assert "grounding_accuracy" in report.quality_regressions
    assert "grounding_accuracy" not in report.quality_improvements


def test_compare_flags_lower_is_better_latency_regression() -> None:
    baseline = build_summary(performance_aggregate={"end_to_end_latency_ms": 1000.0})
    candidate = build_summary(performance_aggregate={"end_to_end_latency_ms": 2000.0})

    report = compare(baseline, candidate)

    assert "end_to_end_latency_ms" in report.latency_regressions


def test_compare_flags_lower_is_better_latency_improvement() -> None:
    baseline = build_summary(performance_aggregate={"end_to_end_latency_ms": 2000.0})
    candidate = build_summary(performance_aggregate={"end_to_end_latency_ms": 1000.0})

    report = compare(baseline, candidate)

    assert "end_to_end_latency_ms" not in report.latency_regressions
    comparison = next(
        c for c in report.performance_comparisons if c.metric_name == "end_to_end_latency_ms"
    )
    assert comparison.direction == "improvement"


def test_compare_classifies_small_changes_as_unchanged() -> None:
    baseline = build_summary(generation_aggregate={"grounding_accuracy": 0.80})
    candidate = build_summary(generation_aggregate={"grounding_accuracy": 0.81})

    report = compare(baseline, candidate)

    comparison = next(
        c for c in report.generation_comparisons if c.metric_name == "grounding_accuracy"
    )
    assert comparison.direction == "unchanged"


def test_compare_handles_zero_baseline_without_dividing_by_zero() -> None:
    baseline = build_summary(generation_aggregate={"unsupported_claim_rate": 0.0})
    candidate = build_summary(generation_aggregate={"unsupported_claim_rate": 0.5})

    report = compare(baseline, candidate)

    comparison = next(
        c for c in report.generation_comparisons if c.metric_name == "unsupported_claim_rate"
    )
    assert comparison.direction == "regression"


def test_compare_reports_dataset_version_mismatch() -> None:
    baseline = build_summary(dataset_version="hash-a")
    candidate = build_summary(dataset_version="hash-b")

    report = compare(baseline, candidate)

    assert report.dataset_versions_match is False


def test_compare_reports_dataset_version_match() -> None:
    baseline = build_summary(dataset_version="hash-a")
    candidate = build_summary(dataset_version="hash-a")

    report = compare(baseline, candidate)

    assert report.dataset_versions_match is True


def test_compare_records_baseline_and_candidate_benchmark_ids() -> None:
    baseline = build_summary(benchmark_id="bench-a")
    candidate = build_summary(benchmark_id="bench-b")

    report = compare(baseline, candidate)

    assert report.baseline_benchmark_id == "bench-a"
    assert report.candidate_benchmark_id == "bench-b"
