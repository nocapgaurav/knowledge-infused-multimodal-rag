"""Unit tests for benchmark artifact persistence.

Deliberately not reusing `backend.storage.WorkspaceStorage` for this test
either -- `EvaluationRepository` is its own standalone, direct
`pathlib.Path`-based implementation, keyed by `benchmark_id`, not `PaperId`.
"""

from pathlib import Path

import pytest

from backend.evaluation.exceptions import BenchmarkNotFoundError, NoBenchmarkRunYetError
from backend.evaluation.repository.evaluation_repository import EvaluationRepository

from ._helpers import build_summary


def test_save_and_load_summary_round_trips(tmp_path: Path) -> None:
    repository = EvaluationRepository(storage_root=tmp_path)
    summary = build_summary(benchmark_id="bench-a")

    repository.save_summary(summary)
    loaded = repository.load_summary("bench-a")

    assert loaded == summary


def test_load_summary_raises_for_unknown_benchmark_id(tmp_path: Path) -> None:
    repository = EvaluationRepository(storage_root=tmp_path)

    with pytest.raises(BenchmarkNotFoundError):
        repository.load_summary("does-not-exist")


def test_load_latest_summary_raises_when_nothing_has_run(tmp_path: Path) -> None:
    repository = EvaluationRepository(storage_root=tmp_path)

    with pytest.raises(NoBenchmarkRunYetError):
        repository.load_latest_summary()


def test_load_latest_summary_returns_the_most_recently_saved_run(tmp_path: Path) -> None:
    repository = EvaluationRepository(storage_root=tmp_path)
    repository.save_summary(build_summary(benchmark_id="bench-a"))
    repository.save_summary(build_summary(benchmark_id="bench-b"))

    latest = repository.load_latest_summary()

    assert latest.manifest.benchmark_id == "bench-b"


def test_save_report_file_and_read_it_back(tmp_path: Path) -> None:
    repository = EvaluationRepository(storage_root=tmp_path)

    repository.save_report_file("bench-a", "benchmark.md", "# Report")

    assert repository.read_report_file("bench-a", "benchmark.md") == "# Report"


def test_read_report_file_returns_none_when_missing(tmp_path: Path) -> None:
    repository = EvaluationRepository(storage_root=tmp_path)

    assert repository.read_report_file("bench-a", "benchmark.md") is None
