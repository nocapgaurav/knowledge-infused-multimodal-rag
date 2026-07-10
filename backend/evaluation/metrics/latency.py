"""Latency: combines externally-measured wall-clock timings into one view.

The actual timing (wrapping `time.perf_counter()` around a real retrieval
or generation call) happens in `benchmark/benchmark_runner.py`, the only
place that calls those services -- this module only combines already-
measured durations, keeping the timing mechanism itself out of a "metric"
file that should stay a pure function.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyMeasurement:
    """Wall-clock timings for one evaluation case.

    Attributes:
        retrieval_latency_ms: Time spent in the retrieval call.
        generation_latency_ms: Time spent in the generation call.
        end_to_end_latency_ms: Total time for this case.
    """

    retrieval_latency_ms: float
    generation_latency_ms: float
    end_to_end_latency_ms: float


def measure_latency(
    retrieval_latency_ms: float, generation_latency_ms: float
) -> LatencyMeasurement:
    """Combine externally-measured retrieval and generation timings.

    Args:
        retrieval_latency_ms: Elapsed time of the retrieval call, in milliseconds.
        generation_latency_ms: Elapsed time of the generation call, in milliseconds.

    Returns:
        The combined latency measurement, including the end-to-end total.
    """
    return LatencyMeasurement(
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        end_to_end_latency_ms=retrieval_latency_ms + generation_latency_ms,
    )
