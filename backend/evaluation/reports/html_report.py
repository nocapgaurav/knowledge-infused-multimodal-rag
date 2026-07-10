"""HTML report: a styled, shareable view of a benchmark run for broader,
non-engineering review.

Renders the same data `markdown_report.py` does, from the same
`EvaluationSummary` -- a different presentation of one source of truth,
never a second computation of it.
"""

from html import escape

from backend.evaluation.benchmark.regression import RegressionReport
from backend.evaluation.models.evaluation_summary import EvaluationSummary
from backend.evaluation.reports.recommendations import generate_recommendations

_STYLE = """
body { font-family: -apple-system, sans-serif; margin: 2rem; color: #1a1a1a; }
h1, h2, h3 { border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }
th, td { border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; font-size: 0.9rem; }
th { background: #f5f5f5; }
.warning { color: #a33; font-weight: bold; }
.section { margin-bottom: 2rem; }
"""


def render_html(summary: EvaluationSummary, regression: RegressionReport | None = None) -> str:
    """Render a benchmark run (and optionally a regression comparison) as HTML.

    Args:
        summary: The complete benchmark run to render.
        regression: An optional regression comparison against a prior run.

    Returns:
        The complete HTML document, suitable for writing to `benchmark.html`.
    """
    body = [
        f"<h1>Benchmark Report: {escape(summary.manifest.benchmark_id)}</h1>",
        _render_overview(summary),
        "<div class='section'><h2>Retrieval Metrics (Dense vs Hybrid)</h2>",
        _render_comparison_table(
            summary.dense_retrieval_aggregate, summary.hybrid_retrieval_aggregate
        ),
        "</div>",
        "<div class='section'><h2>Generation Metrics</h2>",
        _render_metric_table(summary.generation_aggregate),
        "</div>",
        "<div class='section'><h2>Performance Metrics</h2>",
        _render_metric_table(summary.performance_aggregate),
        "</div>",
    ]
    body.append(_render_breakdown("Difficulty", summary.breakdown_by_difficulty))
    body.append(_render_breakdown("Category", summary.breakdown_by_category))
    if regression is not None:
        body.append("<div class='section'><h2>Regression Summary</h2>")
        body.append(_render_regression(regression))
        body.append("</div>")
    body.append("<div class='section'><h2>Case Details</h2>")
    body.append(_render_case_details(summary))
    body.append("</div>")
    body.append("<div class='section'><h2>Recommendations</h2><ul>")
    for recommendation in generate_recommendations(summary):
        body.append(f"<li>{escape(recommendation)}</li>")
    body.append("</ul></div>")

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Benchmark Report</title><style>{_STYLE}</style></head>"
        f"<body>{''.join(body)}</body></html>"
    )


def _render_overview(summary: EvaluationSummary) -> str:
    manifest = summary.manifest
    rows = [
        ("Dataset version", manifest.dataset_version[:16] + "..."),
        ("Dataset case count", str(manifest.dataset_case_count)),
        ("Cases evaluated", str(len(summary.case_results))),
        ("Cases failed", str(len(summary.failed_cases))),
        ("Answer status accuracy", f"{summary.answer_status_accuracy:.2%}"),
        ("Retrieval strategy version", manifest.retrieval_strategy_version or "n/a"),
        ("Generation prompt version", manifest.generation_prompt_version or "n/a"),
        ("Generated at", manifest.created_at.isoformat()),
    ]
    rows_html = "".join(f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>" for k, v in rows)
    return f"<div class='section'><h2>Overall Summary</h2><table>{rows_html}</table></div>"


def _render_metric_table(aggregate: dict[str, float]) -> str:
    if not aggregate:
        return "<p>No cases completed; no metrics available.</p>"
    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{aggregate[name]:.4f}</td></tr>"
        for name in sorted(aggregate)
    )
    return f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"


def _render_comparison_table(dense: dict[str, float], hybrid: dict[str, float]) -> str:
    if not dense and not hybrid:
        return "<p>No cases completed; no metrics available.</p>"
    rows = []
    for name in sorted(set(dense) | set(hybrid)):
        dense_value = f"{dense[name]:.4f}" if name in dense else "n/a"
        hybrid_value = f"{hybrid[name]:.4f}" if name in hybrid else "n/a"
        rows.append(
            f"<tr><td>{escape(name)}</td><td>{dense_value}</td><td>{hybrid_value}</td></tr>"
        )
    return f"<table><tr><th>Metric</th><th>Dense</th><th>Hybrid</th></tr>{''.join(rows)}</table>"


def _render_breakdown(label: str, breakdown: dict[str, dict[str, float]]) -> str:
    if not breakdown:
        return ""
    sections = [f"<div class='section'><h2>Breakdown by {escape(label)}</h2>"]
    for key in sorted(breakdown):
        sections.append(f"<h3>{escape(key)}</h3>{_render_metric_table(breakdown[key])}")
    sections.append("</div>")
    return "".join(sections)


def _render_regression(regression: RegressionReport) -> str:
    parts = [
        f"<p>Baseline: <code>{escape(regression.baseline_benchmark_id)}</code> &rarr; "
        f"Candidate: <code>{escape(regression.candidate_benchmark_id)}</code></p>"
    ]
    if not regression.dataset_versions_match:
        parts.append(
            "<p class='warning'>Warning: dataset versions differ between these two "
            "runs -- this comparison may not be meaningful.</p>"
        )
    parts.append(_render_named_list("Quality Improvements", regression.quality_improvements))
    parts.append(_render_named_list("Quality Regressions", regression.quality_regressions))
    parts.append(_render_named_list("Latency Regressions", regression.latency_regressions))
    return "".join(parts)


def _render_named_list(title: str, names: tuple[str, ...]) -> str:
    items = "".join(f"<li>{escape(name)}</li>" for name in names) if names else "<li>none</li>"
    return f"<h3>{escape(title)}</h3><ul>{items}</ul>"


def _render_case_details(summary: EvaluationSummary) -> str:
    if not summary.case_results:
        return "<p>No cases completed.</p>"
    header = (
        "<tr><th>Case</th><th>Category</th><th>Difficulty</th><th>Status</th>"
        "<th>Expected Status</th><th>Grounding</th><th>Completeness</th></tr>"
    )
    rows = []
    for result in summary.case_results:
        status_marker = "OK" if result.generation_metrics.answer_status_correct else "MISMATCH"
        rows.append(
            "<tr>"
            f"<td>{escape(result.case_id)}</td>"
            f"<td>{escape(result.category)}</td>"
            f"<td>{escape(result.difficulty.value)}</td>"
            f"<td>{escape(result.answer_status.value)}</td>"
            f"<td>{escape(result.expected_answer_status.value)} ({status_marker})</td>"
            f"<td>{result.generation_metrics.grounding_accuracy:.2f}</td>"
            f"<td>{result.generation_metrics.answer_completeness:.2f}</td>"
            "</tr>"
        )
    return f"<table>{header}{''.join(rows)}</table>"
