"""Persists and reads back benchmark artifacts.

Deliberately does not reuse `backend.storage.WorkspaceStorage`: that
abstraction is keyed by `PaperId` -- one workspace per *document*. A
benchmark run spans many documents (one per case), so "one workspace per
benchmark run" is a genuinely different shape, not a document workspace
with a different label. Forcing the fit would produce exactly the kind of
leaky abstraction this project has avoided everywhere else; a small,
direct, benchmark_id-keyed implementation is simpler and more honest.
"""

import json
import logging
from pathlib import Path

from backend.evaluation.exceptions import (
    BenchmarkNotFoundError,
    EvaluationStorageError,
    NoBenchmarkRunYetError,
)
from backend.evaluation.models.evaluation_summary import EvaluationSummary
from backend.evaluation.reports.json_report import render_json

logger = logging.getLogger(__name__)

_SUMMARY_FILENAME = "benchmark.json"
_LATEST_POINTER_FILENAME = "latest.json"


class EvaluationRepository:
    """Reads and writes benchmark artifacts on local disk, keyed by benchmark_id."""

    def __init__(self, storage_root: Path) -> None:
        """Initialize the repository.

        Args:
            storage_root: Base directory benchmark runs are written under.
                Created on first use if it does not already exist.
        """
        self._storage_root = storage_root
        self._storage_root.mkdir(parents=True, exist_ok=True)

    def save_summary(self, summary: EvaluationSummary) -> None:
        """Persist a benchmark run's complete summary and mark it as the latest.

        Args:
            summary: The complete evaluation summary to persist.

        Raises:
            EvaluationStorageError: A storage failure prevented persistence.
        """
        benchmark_id = summary.manifest.benchmark_id
        self.save_report_file(benchmark_id, _SUMMARY_FILENAME, render_json(summary))
        self._mark_latest(benchmark_id)

    def load_summary(self, benchmark_id: str) -> EvaluationSummary:
        """Load a previously persisted benchmark summary by id.

        Args:
            benchmark_id: Identifier of the benchmark run to load.

        Returns:
            The persisted evaluation summary.

        Raises:
            BenchmarkNotFoundError: No benchmark exists with this id.
        """
        content = self.read_report_file(benchmark_id, _SUMMARY_FILENAME)
        if content is None:
            raise BenchmarkNotFoundError(benchmark_id=benchmark_id)
        return EvaluationSummary.model_validate(json.loads(content))

    def load_latest_summary(self) -> EvaluationSummary:
        """Load the most recently persisted benchmark summary.

        Returns:
            The latest evaluation summary.

        Raises:
            NoBenchmarkRunYetError: No benchmark has ever been run.
        """
        pointer_path = self._storage_root / _LATEST_POINTER_FILENAME
        if not pointer_path.is_file():
            raise NoBenchmarkRunYetError()
        try:
            latest_id = json.loads(pointer_path.read_text(encoding="utf-8"))["benchmark_id"]
        except OSError as exc:
            raise NoBenchmarkRunYetError() from exc
        return self.load_summary(latest_id)

    def save_report_file(self, benchmark_id: str, filename: str, content: str) -> None:
        """Persist a report file (JSON, Markdown, or HTML) for a benchmark run.

        Args:
            benchmark_id: Identifier of the benchmark run.
            filename: Name of the file within the benchmark's directory.
            content: File content to write.

        Raises:
            EvaluationStorageError: A storage failure prevented persistence.
        """
        try:
            benchmark_dir = self._benchmark_dir(benchmark_id)
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            (benchmark_dir / filename).write_text(content, encoding="utf-8")
        except OSError as exc:
            raise EvaluationStorageError(benchmark_id=benchmark_id) from exc

    def read_report_file(self, benchmark_id: str, filename: str) -> str | None:
        """Read a previously persisted report file for a benchmark run.

        Args:
            benchmark_id: Identifier of the benchmark run.
            filename: Name of the file within the benchmark's directory.

        Returns:
            The file's content, or `None` if it does not exist.

        Raises:
            EvaluationStorageError: A storage failure prevented the read.
        """
        path = self._benchmark_dir(benchmark_id) / filename
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise EvaluationStorageError(benchmark_id=benchmark_id) from exc

    def _mark_latest(self, benchmark_id: str) -> None:
        pointer_path = self._storage_root / _LATEST_POINTER_FILENAME
        try:
            pointer_path.write_text(json.dumps({"benchmark_id": benchmark_id}), encoding="utf-8")
        except OSError as exc:
            raise EvaluationStorageError(benchmark_id=benchmark_id) from exc

    def _benchmark_dir(self, benchmark_id: str) -> Path:
        return self._storage_root / benchmark_id
