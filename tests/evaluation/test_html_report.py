"""Unit tests for the human-readable HTML report."""

from backend.evaluation.benchmark.regression import compare
from backend.evaluation.reports.html_report import render_html

from ._helpers import build_benchmark_result, build_summary


def test_render_html_includes_the_overview_and_every_section() -> None:
    summary = build_summary(benchmark_id="bench-a")

    rendered = render_html(summary)

    assert "<h1>Benchmark Report: bench-a</h1>" in rendered
    assert "Overall Summary" in rendered
    assert "Retrieval Metrics (Dense vs Hybrid)" in rendered
    assert "Generation Metrics" in rendered
    assert "Performance Metrics" in rendered
    assert "Case Details" in rendered
    assert "Recommendations" in rendered


def test_render_html_omits_regression_summary_when_none_given() -> None:
    summary = build_summary()

    rendered = render_html(summary, regression=None)

    assert "Regression Summary" not in rendered


def test_render_html_includes_regression_summary_when_given() -> None:
    baseline = build_summary(
        benchmark_id="bench-a", generation_aggregate={"grounding_accuracy": 0.5}
    )
    candidate = build_summary(
        benchmark_id="bench-b", generation_aggregate={"grounding_accuracy": 0.9}
    )
    regression = compare(baseline, candidate)

    rendered = render_html(candidate, regression=regression)

    assert "Regression Summary" in rendered
    assert "bench-a" in rendered
    assert "grounding_accuracy" in rendered


def test_render_html_escapes_untrusted_case_text() -> None:
    summary = build_summary(
        case_results=(build_benchmark_result(case_id="<script>alert(1)</script>"),)
    )

    rendered = render_html(summary)

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;" in rendered
