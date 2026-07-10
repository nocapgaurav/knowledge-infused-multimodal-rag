"""Markdown report: a human-readable view of a benchmark run, for
rendering directly in PRs, git diffs, and docs.
"""

from backend.evaluation.benchmark.regression import RegressionReport
from backend.evaluation.models.evaluation_summary import EvaluationSummary
from backend.evaluation.reports.recommendations import generate_recommendations


def render_markdown(summary: EvaluationSummary, regression: RegressionReport | None = None) -> str:
    """Render a benchmark run (and optionally a regression comparison) as Markdown.

    Args:
        summary: The complete benchmark run to render.
        regression: An optional regression comparison against a prior run.

    Returns:
        The Markdown text, suitable for writing to `benchmark.md`.
    """
    lines: list[str] = []
    lines.extend(_render_overview(summary))
    lines.append("")
    lines.append("## Retrieval Metrics (Dense vs Hybrid)")
    lines.append("")
    lines.extend(
        _render_comparison_table(
            summary.dense_retrieval_aggregate, summary.hybrid_retrieval_aggregate
        )
    )
    lines.append("")
    lines.append("## Generation Metrics")
    lines.append("")
    lines.extend(_render_metric_table(summary.generation_aggregate))
    lines.append("")
    lines.append("## Performance Metrics")
    lines.append("")
    lines.extend(_render_metric_table(summary.performance_aggregate))
    lines.append("")
    lines.extend(_render_breakdown("Difficulty", summary.breakdown_by_difficulty))
    lines.extend(_render_breakdown("Category", summary.breakdown_by_category))
    if regression is not None:
        lines.append("## Regression Summary")
        lines.append("")
        lines.extend(_render_regression(regression))
        lines.append("")
    lines.append("## Case Details")
    lines.append("")
    lines.extend(_render_case_details(summary))
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    for recommendation in generate_recommendations(summary):
        lines.append(f"- {recommendation}")
    lines.append("")
    return "\n".join(lines)


def _render_overview(summary: EvaluationSummary) -> list[str]:
    manifest = summary.manifest
    return [
        f"# Benchmark Report: {manifest.benchmark_id}",
        "",
        "## Overall Summary",
        "",
        f"- Dataset version: `{manifest.dataset_version[:16]}...`",
        f"- Dataset case count: {manifest.dataset_case_count}",
        f"- Cases evaluated: {len(summary.case_results)}",
        f"- Cases failed: {len(summary.failed_cases)}",
        f"- Answer status accuracy: {summary.answer_status_accuracy:.2%}",
        f"- Retrieval strategy version: {manifest.retrieval_strategy_version or 'n/a'}",
        f"- Generation prompt version: {manifest.generation_prompt_version or 'n/a'}",
        f"- Generated at: {manifest.created_at.isoformat()}",
    ]


def _render_metric_table(aggregate: dict[str, float]) -> list[str]:
    if not aggregate:
        return ["_No cases completed; no metrics available._"]
    lines = ["| Metric | Value |", "| --- | --- |"]
    for name in sorted(aggregate):
        lines.append(f"| {name} | {aggregate[name]:.4f} |")
    return lines


def _render_comparison_table(dense: dict[str, float], hybrid: dict[str, float]) -> list[str]:
    if not dense and not hybrid:
        return ["_No cases completed; no metrics available._"]
    lines = ["| Metric | Dense | Hybrid |", "| --- | --- | --- |"]
    for name in sorted(set(dense) | set(hybrid)):
        dense_value = f"{dense[name]:.4f}" if name in dense else "n/a"
        hybrid_value = f"{hybrid[name]:.4f}" if name in hybrid else "n/a"
        lines.append(f"| {name} | {dense_value} | {hybrid_value} |")
    return lines


def _render_breakdown(label: str, breakdown: dict[str, dict[str, float]]) -> list[str]:
    if not breakdown:
        return []
    lines = [f"## Breakdown by {label}", ""]
    for key in sorted(breakdown):
        lines.append(f"### {key}")
        lines.append("")
        lines.extend(_render_metric_table(breakdown[key]))
        lines.append("")
    return lines


def _render_regression(regression: RegressionReport) -> list[str]:
    lines = [
        f"Baseline: `{regression.baseline_benchmark_id}` -> "
        f"Candidate: `{regression.candidate_benchmark_id}`",
        "",
    ]
    if not regression.dataset_versions_match:
        lines.append(
            "**Warning: dataset versions differ between these two runs -- this "
            "comparison may not be meaningful.**"
        )
        lines.append("")

    lines.append("### Quality Improvements")
    lines.extend(f"- {name}" for name in regression.quality_improvements)
    if not regression.quality_improvements:
        lines.append("- none")
    lines.append("")

    lines.append("### Quality Regressions")
    lines.extend(f"- {name}" for name in regression.quality_regressions)
    if not regression.quality_regressions:
        lines.append("- none")
    lines.append("")

    lines.append("### Latency Regressions")
    lines.extend(f"- {name}" for name in regression.latency_regressions)
    if not regression.latency_regressions:
        lines.append("- none")
    return lines


def _render_case_details(summary: EvaluationSummary) -> list[str]:
    if not summary.case_results:
        return ["_No cases completed._"]
    lines = [
        "| Case | Category | Difficulty | Status | Expected | Grounding | Completeness |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary.case_results:
        marker = "OK" if result.generation_metrics.answer_status_correct else "MISMATCH"
        expected = f"{result.expected_answer_status.value} ({marker})"
        grounding = result.generation_metrics.grounding_accuracy
        completeness = result.generation_metrics.answer_completeness
        lines.append(
            f"| {result.case_id} | {result.category} | {result.difficulty.value} | "
            f"{result.answer_status.value} | {expected} | {grounding:.2f} | {completeness:.2f} |"
        )
    return lines
