"""Tests for the health check endpoint."""

from fastapi.testclient import TestClient

from backend.api.app import create_app


def test_health_returns_ok() -> None:
    """The health endpoint should report a healthy status."""
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
