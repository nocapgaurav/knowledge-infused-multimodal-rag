"""Evaluation suite's own data models -- not part of `backend.domain` for
the same reason every prior infrastructure module's models aren't: these
describe versioned benchmark output, not permanent facts about a paper.
"""

from backend.evaluation.models.benchmark_manifest import BenchmarkManifest
from backend.evaluation.models.benchmark_result import (
    BenchmarkResult,
    GenerationCaseMetrics,
    PerformanceCaseMetrics,
    RetrievalCaseMetrics,
)
from backend.evaluation.models.evaluation_case import Difficulty, EvaluationCase
from backend.evaluation.models.evaluation_summary import CaseFailure, EvaluationSummary

__all__ = [
    "BenchmarkManifest",
    "BenchmarkResult",
    "CaseFailure",
    "Difficulty",
    "EvaluationCase",
    "EvaluationSummary",
    "GenerationCaseMetrics",
    "PerformanceCaseMetrics",
    "RetrievalCaseMetrics",
]
