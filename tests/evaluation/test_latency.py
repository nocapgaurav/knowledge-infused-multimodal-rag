"""Unit tests for combining externally-measured latency timings."""

from backend.evaluation.metrics.latency import measure_latency


def test_measure_latency_sums_retrieval_and_generation_into_end_to_end() -> None:
    measurement = measure_latency(retrieval_latency_ms=120.0, generation_latency_ms=880.0)

    assert measurement.retrieval_latency_ms == 120.0
    assert measurement.generation_latency_ms == 880.0
    assert measurement.end_to_end_latency_ms == 1000.0


def test_measure_latency_of_zero_durations_is_zero() -> None:
    measurement = measure_latency(retrieval_latency_ms=0.0, generation_latency_ms=0.0)

    assert measurement.end_to_end_latency_ms == 0.0
