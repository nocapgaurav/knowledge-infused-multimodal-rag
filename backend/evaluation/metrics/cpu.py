"""CPU usage: process user+system CPU time consumed during one operation.

Unlike `memory.py`'s watermark, `ru_utime`/`ru_stime` are cumulative
counters that only increase -- a before/after snapshot correctly isolates
the CPU time consumed between the two snapshots.
"""

import resource
from dataclasses import dataclass


@dataclass(frozen=True)
class CpuSnapshot:
    """A point-in-time reading of cumulative process CPU time.

    Attributes:
        user_seconds: Cumulative user-mode CPU time since process start.
        system_seconds: Cumulative system-mode CPU time since process start.
    """

    user_seconds: float
    system_seconds: float


def snapshot_cpu_time() -> CpuSnapshot:
    """Take a CPU time snapshot for the current process.

    Returns:
        The current cumulative user and system CPU time.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return CpuSnapshot(user_seconds=usage.ru_utime, system_seconds=usage.ru_stime)


def cpu_time_delta_ms(before: CpuSnapshot, after: CpuSnapshot) -> float:
    """Compute CPU time consumed between two snapshots.

    Args:
        before: Snapshot taken before the operation.
        after: Snapshot taken after the operation.

    Returns:
        Total user+system CPU time consumed, in milliseconds.
    """
    delta_seconds = (after.user_seconds - before.user_seconds) + (
        after.system_seconds - before.system_seconds
    )
    return delta_seconds * 1000
