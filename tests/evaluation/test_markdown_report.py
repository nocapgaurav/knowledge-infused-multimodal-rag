"""Unit tests for the human-readable Markdown report."""

from backend.evaluation.benchmark.regression import compare
from backend.evaluation.reports.markdown_report import render_markdown

from ._helpers import build_benchmark_result, build_summary


def test_render_markdown_includes_the_overview_and_every_section() -> None:
    summary = build_summary(benchmark_id="bench-a")

    rendered = render_markdown(summary)

    assert "# Benchmark Report: bench-a" in rendered
    assert "## Overall Summary" in rendered
    assert "## Retrieval Metrics (Dense vs Hybrid)" in rendered
    assert "## Generation Metrics" in rendered
    assert "## Performance Metrics" in rendered
    assert "## Case Details" in rendered
    assert "## Recommendations" in rendered


def test_render_markdown_omits_regression_summary_when_none_given() -> None:
    summary = build_summary()

    rendered = render_markdown(summary, regression=None)

    assert "## Regression Summary" not in rendered


def test_render_markdown_includes_regression_summary_when_given() -> None:
    baseline = build_summary(
        benchmark_id="bench-a", generation_aggregate={"grounding_accuracy": 0.5}
    )
    candidate = build_summary(
        benchmark_id="bench-b", generation_aggregate={"grounding_accuracy": 0.9}
    )
    regression = compare(baseline, candidate)

    rendered = render_markdown(candidate, regression=regression)

    assert "## Regression Summary" in rendered
    assert "bench-a" in rendered
    assert "bench-b" in rendered
    assert "grounding_accuracy" in rendered


def test_render_markdown_lists_every_case_by_id() -> None:
    summary = build_summary(case_results=(build_benchmark_result(case_id="case-xyz"),))

    rendered = render_markdown(summary)

    assert "case-xyz" in rendered
