"""API routes for the evaluation and validation suite.

Both routes are thin: `POST /evaluation/run` runs the real benchmark suite
against Modules 1-10's own production services and returns the resulting
summary directly, and `GET /evaluation/report` returns the most recently
persisted summary. Neither route touches retrieval or generation logic
itself -- that composition lives entirely inside `EvaluationService` and
the modules it calls.
"""

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_evaluation_service
from backend.evaluation.models.evaluation_summary import EvaluationSummary
from backend.evaluation.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationSummary)
async def run_benchmark(
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationSummary:
    """Run the complete benchmark suite against the real pipeline.

    Args:
        evaluation_service: Evaluation service, injected.

    Returns:
        The complete evaluation summary for this run.
    """
    return evaluation_service.run_benchmark()


@router.get("/report", response_model=EvaluationSummary)
async def get_latest_report(
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationSummary:
    """Return the most recently persisted benchmark summary.

    Args:
        evaluation_service: Evaluation service, injected.

    Returns:
        The latest evaluation summary.
    """
    return evaluation_service.get_latest_report()
