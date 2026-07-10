"""Unit tests for CPU time delta measurement."""

import pytest

from backend.evaluation.metrics.cpu import CpuSnapshot, cpu_time_delta_ms, snapshot_cpu_time


def test_cpu_time_delta_ms_sums_user_and_system_time() -> None:
    before = CpuSnapshot(user_seconds=1.0, system_seconds=0.5)
    after = CpuSnapshot(user_seconds=1.2, system_seconds=0.6)

    assert cpu_time_delta_ms(before, after) == pytest.approx(300.0)


def test_cpu_time_delta_ms_of_identical_snapshots_is_zero() -> None:
    snapshot = CpuSnapshot(user_seconds=2.0, system_seconds=1.0)

    assert cpu_time_delta_ms(snapshot, snapshot) == 0.0


def test_snapshot_cpu_time_is_non_negative() -> None:
    snapshot = snapshot_cpu_time()

    assert snapshot.user_seconds >= 0.0
    assert snapshot.system_seconds >= 0.0
