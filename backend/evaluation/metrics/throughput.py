"""Throughput: cases processed per second, across a whole benchmark run.

Computed once per run, not per case -- a single case's "throughput" isn't
a meaningful concept on its own.
"""

from backend.evaluation.exceptions import MetricComputationError


def compute_throughput(case_count: int, total_duration_seconds: float) -> float:
    """Compute throughput for a completed benchmark run.

    Args:
        case_count: Number of cases completed.
        total_duration_seconds: Total wall-clock time for the run.

    Returns:
        Cases processed per second.

    Raises:
        MetricComputationError: `total_duration_seconds` is not positive.
    """
    if total_duration_seconds <= 0:
        raise MetricComputationError(
            reason=f"total_duration_seconds must be positive, got {total_duration_seconds}"
        )
    return case_count / total_duration_seconds
