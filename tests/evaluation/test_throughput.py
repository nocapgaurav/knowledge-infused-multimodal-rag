"""Unit tests for run-level throughput."""

import pytest

from backend.evaluation.exceptions import MetricComputationError
from backend.evaluation.metrics.throughput import compute_throughput


def test_compute_throughput_divides_cases_by_duration() -> None:
    assert compute_throughput(case_count=10, total_duration_seconds=5.0) == pytest.approx(2.0)


def test_compute_throughput_rejects_non_positive_duration() -> None:
    with pytest.raises(MetricComputationError):
        compute_throughput(case_count=10, total_duration_seconds=0.0)
