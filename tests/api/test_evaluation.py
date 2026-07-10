"""End-to-end tests for the evaluation and validation suite API.

Overrides `EvaluationService` with a fake -- this verifies routing,
dependency wiring, and status-code mapping, not the real pipeline itself
(covered separately in
tests/evaluation/test_evaluation_pipeline_integration.py's real-stack case).
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import get_evaluation_service
from backend.evaluation.exceptions import (
    DatasetNotFoundError,
    DuplicateCaseIdError,
    EvaluationStorageError,
    NoBenchmarkRunYetError,
)
from backend.evaluation.models.evaluation_summary import EvaluationSummary
from tests.evaluation._helpers import build_summary


class _FakeEvaluationService:
    def __init__(
        self,
        *,
        summary: EvaluationSummary | None = None,
        run_error: Exception | None = None,
        report_error: Exception | None = None,
    ) -> None:
        self._summary = summary
        self._run_error = run_error
        self._report_error = report_error

    def run_benchmark(self) -> EvaluationSummary:
        if self._run_error is not None:
            raise self._run_error
        return self._summary or build_summary()

    def get_latest_report(self) -> EvaluationSummary:
        if self._report_error is not None:
            raise self._report_error
        return self._summary or build_summary()


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_evaluation_service] = lambda: _FakeEvaluationService()
    with TestClient(app) as test_client:
        yield test_client


def test_run_benchmark_returns_the_evaluation_summary(client: TestClient) -> None:
    response = client.post("/evaluation/run")

    assert response.status_code == 200
    body = response.json()
    assert body["manifest"]["benchmark_id"] == "benchmark-1"
    assert len(body["case_results"]) == 1


def test_get_latest_report_returns_the_evaluation_summary(client: TestClient) -> None:
    response = client.get("/evaluation/report")

    assert response.status_code == 200
    assert response.json()["manifest"]["benchmark_id"] == "benchmark-1"


def test_get_latest_report_returns_404_when_no_benchmark_has_run() -> None:
    app = create_app()
    app.dependency_overrides[get_evaluation_service] = lambda: _FakeEvaluationService(
        report_error=NoBenchmarkRunYetError()
    )
    with TestClient(app) as client:
        response = client.get("/evaluation/report")

    assert response.status_code == 404


def test_run_benchmark_returns_404_when_dataset_file_missing() -> None:
    app = create_app()
    app.dependency_overrides[get_evaluation_service] = lambda: _FakeEvaluationService(
        run_error=DatasetNotFoundError(path="data/evaluation_dataset.json")
    )
    with TestClient(app) as client:
        response = client.post("/evaluation/run")

    assert response.status_code == 404


def test_run_benchmark_returns_422_on_dataset_validation_error() -> None:
    app = create_app()
    app.dependency_overrides[get_evaluation_service] = lambda: _FakeEvaluationService(
        run_error=DuplicateCaseIdError(case_id="case-001")
    )
    with TestClient(app) as client:
        response = client.post("/evaluation/run")

    assert response.status_code == 422


def test_run_benchmark_returns_500_on_storage_failure() -> None:
    app = create_app()
    app.dependency_overrides[get_evaluation_service] = lambda: _FakeEvaluationService(
        run_error=EvaluationStorageError(benchmark_id="bench-x")
    )
    with TestClient(app) as client:
        response = client.post("/evaluation/run")

    assert response.status_code == 500
