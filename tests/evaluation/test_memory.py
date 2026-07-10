"""Unit tests for peak process memory measurement.

Only asserts what is portably true of `ru_maxrss` on any platform: it is
a positive, monotonically non-decreasing watermark -- not an exact value,
since the raw unit (bytes on macOS, kilobytes on Linux) is platform
-dependent and already handled inside `current_peak_memory_mb` itself.
"""

from backend.evaluation.metrics.memory import current_peak_memory_mb


def test_current_peak_memory_mb_is_positive() -> None:
    assert current_peak_memory_mb() > 0.0


def test_current_peak_memory_mb_never_decreases() -> None:
    first = current_peak_memory_mb()
    _ = bytearray(10 * 1024 * 1024)
    second = current_peak_memory_mb()

    assert second >= first
