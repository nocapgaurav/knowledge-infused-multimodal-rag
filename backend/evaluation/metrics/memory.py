"""Memory usage: current process peak RSS.

Uses only the standard library (`resource`) -- no new third-party
dependency for something the platform already provides. `ru_maxrss` is a
watermark (the peak RSS reached since process start), not a per-operation
delta: it never decreases, so this reports "peak memory so far," not
"memory used by just this case" (see the final report's known limitations).

Confirmed by real measurement on this development machine: macOS reports
`ru_maxrss` in bytes; Linux reports it in kilobytes. Handled portably
rather than assumed.
"""

import resource
import sys

_BYTES_PER_MEGABYTE = 1024 * 1024
_KILOBYTES_PER_MEGABYTE = 1024


def current_peak_memory_mb() -> float:
    """Return the current process's peak RSS, in megabytes.

    Returns:
        Peak resident set size since process start, in MB.
    """
    raw_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw_maxrss / _BYTES_PER_MEGABYTE
    return raw_maxrss / _KILOBYTES_PER_MEGABYTE
