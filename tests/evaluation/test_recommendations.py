"""Unit tests for deterministic, threshold-based recommendations."""

from backend.evaluation.models.evaluation_summary import CaseFailure
from backend.evaluation.reports.recommendations import generate_recommendations

from ._helpers import build_summary


def test_generate_recommendations_flags_low_grounding_accuracy() -> None:
    summary = build_summary(generation_aggregate={"grounding_accuracy": 0.5})

    recommendations = generate_recommendations(summary)

    assert any("grounding accuracy" in r.lower() for r in recommendations)


def test_generate_recommendations_flags_high_unsupported_claim_rate() -> None:
    summary = build_summary(generation_aggregate={"unsupported_claim_rate": 0.5})

    recommendations = generate_recommendations(summary)

    assert any("unsupported claim" in r.lower() for r in recommendations)


def test_generate_recommendations_flags_low_answer_status_accuracy() -> None:
    summary = build_summary(answer_status_accuracy=0.5)

    recommendations = generate_recommendations(summary)

    assert any("answer status accuracy" in r.lower() for r in recommendations)


def test_generate_recommendations_flags_low_citation_accuracy() -> None:
    summary = build_summary(generation_aggregate={"citation_accuracy": 0.5})

    recommendations = generate_recommendations(summary)

    assert any("citation accuracy" in r.lower() for r in recommendations)


def test_generate_recommendations_flags_high_latency() -> None:
    summary = build_summary(performance_aggregate={"end_to_end_latency_ms": 20_000.0})

    recommendations = generate_recommendations(summary)

    assert any("latency" in r.lower() for r in recommendations)


def test_generate_recommendations_flags_failed_cases() -> None:
    summary = build_summary(failed_cases=(CaseFailure(case_id="case-002", reason="timeout"),))

    recommendations = generate_recommendations(summary)

    assert any("failed to complete" in r.lower() for r in recommendations)


def test_generate_recommendations_is_never_empty_when_all_metrics_are_healthy() -> None:
    summary = build_summary()

    recommendations = generate_recommendations(summary)

    assert recommendations == ["All monitored metrics are within expected thresholds."]
