"""Exceptions raised by the evaluation suite.

Notably absent: an exception for "a case's quality metrics were low" or
"a regression was detected." Those are normal, expected outcomes this
module exists to measure and report, never a failure to raise about. Also
notably absent: a dedicated "document not indexed" exception -- Module
9's own `DocumentNotIndexedError`/`DocumentNotGraphedError` already cover
that; the benchmark runner catches them per-case (see `CaseFailure`)
rather than this module inventing a parallel type. Exceptions here are
reserved for the evaluation process itself being unable to proceed -- a
missing dataset, a malformed case, an invalid metric input, or a storage
failure.
"""


class EvaluationError(Exception):
    """Base class for all evaluation suite errors."""


class DatasetNotFoundError(EvaluationError):
    """Raised when the evaluation dataset file does not exist."""

    def __init__(self, *, path: str) -> None:
        self.path = path
        super().__init__(f"evaluation dataset not found: {path}")


class DatasetValidationError(EvaluationError):
    """Base class for structural defects in an evaluation dataset."""


class EmptyDatasetError(DatasetValidationError):
    """Raised when a dataset contains no cases."""

    def __init__(self, *, path: str) -> None:
        self.path = path
        super().__init__(f"evaluation dataset is empty: {path}")


class DuplicateCaseIdError(DatasetValidationError):
    """Raised when the same case_id appears more than once in a dataset."""

    def __init__(self, *, case_id: str) -> None:
        self.case_id = case_id
        super().__init__(f"duplicate case_id in evaluation dataset: {case_id}")


class MetricComputationError(EvaluationError):
    """Raised when a metric receives an invalid input (e.g. a non-positive k)."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"metric computation failed: {reason}")


class BenchmarkNotFoundError(EvaluationError):
    """Raised when no benchmark result exists for a requested lookup."""

    def __init__(self, *, benchmark_id: str) -> None:
        self.benchmark_id = benchmark_id
        super().__init__(f"no benchmark found with id: {benchmark_id}")


class NoBenchmarkRunYetError(EvaluationError):
    """Raised when the latest report is requested but no benchmark has ever run."""

    def __init__(self) -> None:
        super().__init__("no benchmark has been run yet")


class EvaluationStorageError(EvaluationError):
    """Raised when a storage failure prevents benchmark artifacts from being persisted."""

    def __init__(self, *, benchmark_id: str) -> None:
        self.benchmark_id = benchmark_id
        super().__init__(f"a storage error occurred while persisting benchmark {benchmark_id}")
