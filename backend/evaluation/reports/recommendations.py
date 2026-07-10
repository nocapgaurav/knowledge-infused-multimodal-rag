"""Deterministic, threshold-based recommendations derived from a benchmark
run -- shared by every report format so recommendations never diverge
between them.

Every threshold is named and justified below, never an unexplained magic
number, and no threshold check is a substitute for human review -- these
are conservative, generic flags, not authoritative verdicts.
"""

from backend.evaluation.models.evaluation_summary import EvaluationSummary

_LOW_GROUNDING_ACCURACY_THRESHOLD = 0.7
_HIGH_UNSUPPORTED_CLAIM_RATE_THRESHOLD = 0.3
_LOW_ANSWER_STATUS_ACCURACY_THRESHOLD = 0.8
_LOW_CITATION_ACCURACY_THRESHOLD = 0.7
_HIGH_END_TO_END_LATENCY_MS_THRESHOLD = 15_000.0
"""Generous relative to real Ollama latency observed during Module 10's
own end-to-end verification (roughly 5-12 seconds per call) -- flags a
genuine slowdown, not normal local-LLM variance."""


def generate_recommendations(summary: EvaluationSummary) -> list[str]:
    """Generate deterministic recommendations from a benchmark run's results.

    Args:
        summary: The complete benchmark run to derive recommendations from.

    Returns:
        Human-readable recommendation strings. Always non-empty -- a run
        with nothing to flag still reports that explicitly.
    """
    recommendations: list[str] = []

    if summary.failed_cases:
        recommendations.append(
            f"{len(summary.failed_cases)} case(s) failed to complete -- investigate "
            f"pipeline errors before trusting these results."
        )

    grounding = summary.generation_aggregate.get("grounding_accuracy")
    if grounding is not None and grounding < _LOW_GROUNDING_ACCURACY_THRESHOLD:
        recommendations.append(
            f"Grounding accuracy ({grounding:.2f}) is below {_LOW_GROUNDING_ACCURACY_THRESHOLD} "
            f"-- review prompt grounding rules or context optimization."
        )

    unsupported = summary.generation_aggregate.get("unsupported_claim_rate")
    if unsupported is not None and unsupported > _HIGH_UNSUPPORTED_CLAIM_RATE_THRESHOLD:
        recommendations.append(
            f"Unsupported claim rate ({unsupported:.2f}) exceeds "
            f"{_HIGH_UNSUPPORTED_CLAIM_RATE_THRESHOLD} -- high hallucination risk."
        )

    if summary.answer_status_accuracy < _LOW_ANSWER_STATUS_ACCURACY_THRESHOLD:
        recommendations.append(
            f"Answer status accuracy ({summary.answer_status_accuracy:.2f}) is below "
            f"{_LOW_ANSWER_STATUS_ACCURACY_THRESHOLD} -- review answer planning and "
            f"quality assessment thresholds."
        )

    citation = summary.generation_aggregate.get("citation_accuracy")
    if citation is not None and citation < _LOW_CITATION_ACCURACY_THRESHOLD:
        recommendations.append(
            f"Citation accuracy ({citation:.2f}) is below {_LOW_CITATION_ACCURACY_THRESHOLD} "
            f"-- the model may be inventing or mistyping citation labels."
        )

    latency = summary.performance_aggregate.get("end_to_end_latency_ms")
    if latency is not None and latency > _HIGH_END_TO_END_LATENCY_MS_THRESHOLD:
        recommendations.append(
            f"End-to-end latency ({latency:.0f}ms) exceeds "
            f"{_HIGH_END_TO_END_LATENCY_MS_THRESHOLD:.0f}ms -- consider a smaller or "
            f"faster model, or a smaller context window."
        )

    if not recommendations:
        recommendations.append("All monitored metrics are within expected thresholds.")
    return recommendations
