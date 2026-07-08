"""Health check endpoint for liveness and readiness probes."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Response body for the health check endpoint.

    Attributes:
        status: Liveness status of the service. Always ``"ok"`` when the
            process is able to respond to requests.
    """

    status: Literal["ok"] = "ok"


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Report that the service process is up and able to serve requests.

    Returns:
        A :class:`HealthResponse` indicating the service is healthy.
    """
    return HealthResponse()
