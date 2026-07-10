"""Orchestrates one complete evaluation run.

Load dataset -> validate -> run every case through the real pipeline ->
aggregate -> compare against the previous run, if one exists -> persist
the manifest and all three report formats. Each step (dataset loader,
validator, runner, regression, report renderers) is independently
testable; this class's only job is calling them in the right order.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from backend.evaluation.benchmark.benchmark_runner import BenchmarkRunner
from backend.evaluation.benchmark.regression import RegressionReport, compare
from backend.evaluation.datasets.evaluation_dataset import load_dataset_from_json
from backend.evaluation.exceptions import NoBenchmarkRunYetError
from backend.evaluation.metrics.throughput import compute_throughput
from backend.evaluation.models.benchmark_manifest import BenchmarkManifest
from backend.evaluation.models.evaluation_summary import EvaluationSummary
from backend.evaluation.reports.html_report import render_html
from backend.evaluation.reports.markdown_report import render_markdown
from backend.evaluation.repository.evaluation_repository import EvaluationRepository
from backend.evaluation.validation.evaluation_validator import EvaluationValidator

logger = logging.getLogger(__name__)

BENCHMARK_ARTIFACT_VERSION = "1.0"
EVALUATION_STRATEGY_VERSION = "1.0"
"""Version of this module's own metric-computation and aggregation rules
-- bumped when a metric's definition or aggregation strategy changes,
independently of the manifest schema."""

_MARKDOWN_FILENAME = "benchmark.md"
_HTML_FILENAME = "benchmark.html"


class EvaluationService:
    """Runs the complete evaluation suite for one dataset."""

    def __init__(
        self,
        repository: EvaluationRepository,
        validator: EvaluationValidator,
        runner: BenchmarkRunner,
        dataset_path: Path,
    ) -> None:
        """Initialize the service.

        Args:
            repository: Persists and reads back benchmark artifacts.
            validator: Validates the loaded dataset before running it.
            runner: Runs the dataset through the real pipeline.
            dataset_path: Path to the evaluation dataset JSON file.
        """
        self._repository = repository
        self._validator = validator
        self._runner = runner
        self._dataset_path = dataset_path

    def run_benchmark(self) -> EvaluationSummary:
        """Run one complete benchmark and persist its results and reports.

        If a previous benchmark run exists, this run is automatically
        compared against it and the resulting regression report is
        embedded in the Markdown and HTML reports.

        Returns:
            The complete evaluation summary for this run.

        Raises:
            DatasetNotFoundError: The dataset file does not exist.
            EmptyDatasetError: The dataset contains no cases.
            DuplicateCaseIdError: The dataset has a duplicate case_id.
            EvaluationStorageError: A storage failure prevented persistence.
        """
        dataset = load_dataset_from_json(self._dataset_path)
        self._validator.validate_dataset(dataset, str(self._dataset_path))

        previous_summary = self._load_previous_summary()

        run_result = self._runner.run(dataset.cases)

        performance_aggregate = dict(run_result.performance_aggregate)
        if run_result.case_results and run_result.total_duration_seconds > 0:
            performance_aggregate["throughput_cases_per_second"] = compute_throughput(
                len(run_result.case_results), run_result.total_duration_seconds
            )

        manifest = BenchmarkManifest(
            benchmark_id=str(uuid4()),
            benchmark_version=BENCHMARK_ARTIFACT_VERSION,
            evaluation_strategy_version=EVALUATION_STRATEGY_VERSION,
            dataset_version=dataset.dataset_version,
            dataset_case_count=len(dataset.cases),
            retrieval_strategy_version=run_result.retrieval_strategy_version,
            generation_prompt_version=run_result.generation_prompt_version,
            created_at=datetime.now(UTC),
        )
        summary = EvaluationSummary(
            manifest=manifest,
            case_results=tuple(run_result.case_results),
            failed_cases=tuple(run_result.failed_cases),
            dense_retrieval_aggregate=run_result.dense_retrieval_aggregate,
            hybrid_retrieval_aggregate=run_result.hybrid_retrieval_aggregate,
            generation_aggregate=run_result.generation_aggregate,
            performance_aggregate=performance_aggregate,
            answer_status_accuracy=run_result.answer_status_accuracy,
            breakdown_by_difficulty=run_result.breakdown_by_difficulty,
            breakdown_by_category=run_result.breakdown_by_category,
        )

        regression_report = (
            compare(previous_summary, summary) if previous_summary is not None else None
        )

        self._persist(summary, regression_report)

        logger.info(
            "benchmark run complete",
            extra={
                "benchmark_id": manifest.benchmark_id,
                "cases_completed": len(summary.case_results),
                "cases_failed": len(summary.failed_cases),
            },
        )
        return summary

    def get_latest_report(self) -> EvaluationSummary:
        """Return the most recently persisted benchmark summary.

        Returns:
            The latest evaluation summary.

        Raises:
            NoBenchmarkRunYetError: No benchmark has ever been run.
        """
        return self._repository.load_latest_summary()

    def _load_previous_summary(self) -> EvaluationSummary | None:
        try:
            return self._repository.load_latest_summary()
        except NoBenchmarkRunYetError:
            return None

    def _persist(
        self, summary: EvaluationSummary, regression_report: RegressionReport | None
    ) -> None:
        self._repository.save_summary(summary)
        self._repository.save_report_file(
            summary.manifest.benchmark_id,
            _MARKDOWN_FILENAME,
            render_markdown(summary, regression_report),
        )
        self._repository.save_report_file(
            summary.manifest.benchmark_id,
            _HTML_FILENAME,
            render_html(summary, regression_report),
        )
