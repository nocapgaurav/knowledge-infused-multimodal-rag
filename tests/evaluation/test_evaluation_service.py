"""Unit tests for `EvaluationService`'s orchestration: load -> validate ->
run -> aggregate -> compare against the previous run -> persist.

Uses a fake `BenchmarkRunner` (a test double implementing `.run(cases)`)
so this file exercises only the orchestration itself, not retrieval or
generation -- those are covered by `test_benchmark_runner.py` and the real
end-to-end integration test.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from backend.evaluation.benchmark.benchmark_runner import BenchmarkRunner, BenchmarkRunResult
from backend.evaluation.exceptions import DatasetNotFoundError, EmptyDatasetError
from backend.evaluation.models.evaluation_case import EvaluationCase
from backend.evaluation.repository.evaluation_repository import EvaluationRepository
from backend.evaluation.services.evaluation_service import EvaluationService
from backend.evaluation.validation.evaluation_validator import EvaluationValidator

from ._helpers import build_benchmark_result

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class _FakeBenchmarkRunner(BenchmarkRunner):
    """A real `BenchmarkRunner` subclass that overrides `run` and
    deliberately skips the parent `__init__` -- no real retrieval or
    generation service is ever constructed."""

    def __init__(self, *, grounding_accuracy: float = 0.9) -> None:
        self._grounding_accuracy = grounding_accuracy
        self.received_cases: list[EvaluationCase] | None = None

    def run(self, cases: Sequence[EvaluationCase]) -> BenchmarkRunResult:
        self.received_cases = list(cases)
        result = build_benchmark_result()
        return BenchmarkRunResult(
            case_results=[result],
            failed_cases=[],
            dense_retrieval_aggregate={"mrr": 0.5},
            hybrid_retrieval_aggregate={"mrr": 1.0},
            generation_aggregate={"grounding_accuracy": self._grounding_accuracy},
            performance_aggregate={"end_to_end_latency_ms": 1000.0},
            answer_status_accuracy=1.0,
            breakdown_by_difficulty={"easy": {"grounding_accuracy": self._grounding_accuracy}},
            breakdown_by_category={"factual": {"grounding_accuracy": self._grounding_accuracy}},
            total_duration_seconds=2.0,
            retrieval_strategy_version="1.0",
            generation_prompt_version="1.0",
        )


def _service(
    tmp_path: Path, *, runner: _FakeBenchmarkRunner | None = None, dataset_path: Path | None = None
) -> tuple[EvaluationService, EvaluationRepository]:
    repository = EvaluationRepository(storage_root=tmp_path / "storage")
    service = EvaluationService(
        repository=repository,
        validator=EvaluationValidator(),
        runner=runner or _FakeBenchmarkRunner(),
        dataset_path=dataset_path or (FIXTURES_DIR / "valid_dataset.json"),
    )
    return service, repository


def test_run_benchmark_returns_and_persists_a_summary(tmp_path: Path) -> None:
    service, repository = _service(tmp_path)

    summary = service.run_benchmark()

    assert summary.manifest.dataset_case_count == 2
    persisted = repository.load_summary(summary.manifest.benchmark_id)
    assert persisted == summary


def test_run_benchmark_passes_every_loaded_case_to_the_runner(tmp_path: Path) -> None:
    runner = _FakeBenchmarkRunner()
    service, _ = _service(tmp_path, runner=runner)

    service.run_benchmark()

    assert runner.received_cases is not None
    assert len(runner.received_cases) == 2


def test_run_benchmark_adds_throughput_to_performance_aggregate(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)

    summary = service.run_benchmark()

    assert "throughput_cases_per_second" in summary.performance_aggregate
    assert summary.performance_aggregate["throughput_cases_per_second"] == pytest.approx(0.5)


def test_run_benchmark_persists_markdown_and_html_reports(tmp_path: Path) -> None:
    service, repository = _service(tmp_path)

    summary = service.run_benchmark()

    markdown = repository.read_report_file(summary.manifest.benchmark_id, "benchmark.md")
    html = repository.read_report_file(summary.manifest.benchmark_id, "benchmark.html")
    assert markdown is not None and "# Benchmark Report" in markdown
    assert html is not None and "<h1>Benchmark Report" in html


def test_run_benchmark_omits_regression_on_the_first_run(tmp_path: Path) -> None:
    service, repository = _service(tmp_path)

    summary = service.run_benchmark()

    markdown = repository.read_report_file(summary.manifest.benchmark_id, "benchmark.md")
    assert markdown is not None
    assert "## Regression Summary" not in markdown


def test_run_benchmark_embeds_regression_against_the_previous_run(tmp_path: Path) -> None:
    repository = EvaluationRepository(storage_root=tmp_path / "storage")
    first_service = EvaluationService(
        repository=repository,
        validator=EvaluationValidator(),
        runner=_FakeBenchmarkRunner(grounding_accuracy=0.5),
        dataset_path=FIXTURES_DIR / "valid_dataset.json",
    )
    second_service = EvaluationService(
        repository=repository,
        validator=EvaluationValidator(),
        runner=_FakeBenchmarkRunner(grounding_accuracy=0.9),
        dataset_path=FIXTURES_DIR / "valid_dataset.json",
    )

    first_service.run_benchmark()
    second_summary = second_service.run_benchmark()

    markdown = repository.read_report_file(second_summary.manifest.benchmark_id, "benchmark.md")
    assert markdown is not None
    assert "## Regression Summary" in markdown
    assert "grounding_accuracy" in markdown


def test_get_latest_report_returns_the_most_recent_run(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    service.run_benchmark()
    second = service.run_benchmark()

    latest = service.get_latest_report()

    assert latest.manifest.benchmark_id == second.manifest.benchmark_id


def test_run_benchmark_raises_when_dataset_file_missing(tmp_path: Path) -> None:
    service, _ = _service(tmp_path, dataset_path=tmp_path / "missing.json")

    with pytest.raises(DatasetNotFoundError):
        service.run_benchmark()


def test_run_benchmark_raises_on_empty_dataset(tmp_path: Path) -> None:
    empty_dataset_path = tmp_path / "empty.json"
    empty_dataset_path.write_text(json.dumps([]), encoding="utf-8")
    service, _ = _service(tmp_path, dataset_path=empty_dataset_path)

    with pytest.raises(EmptyDatasetError):
        service.run_benchmark()
