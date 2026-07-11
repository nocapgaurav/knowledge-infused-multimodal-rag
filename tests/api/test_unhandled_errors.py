"""Tests for the unhandled-error response middleware.

Verifies the exact failure mode observed live in Module 12 verification:
an unhandled exception must reach the browser as a readable JSON 500
*with CORS headers*, not as an anonymous response the browser blocks and
misreports as a network failure.
"""

from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.config.settings import get_settings

_FRONTEND_ORIGIN = "http://localhost:3000"


def _client_with_crashing_route() -> TestClient:
    app = create_app()

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("simulated unhandled defect")

    return TestClient(app, raise_server_exceptions=False)


def test_unhandled_error_returns_json_500() -> None:
    client = _client_with_crashing_route()

    response = client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}


def test_unhandled_error_response_carries_cors_headers() -> None:
    assert _FRONTEND_ORIGIN in get_settings().cors_allowed_origins
    client = _client_with_crashing_route()

    response = client.get("/boom", headers={"Origin": _FRONTEND_ORIGIN})

    assert response.status_code == 500
    assert response.headers.get("access-control-allow-origin") == _FRONTEND_ORIGIN


def test_handled_exceptions_still_use_their_registered_handlers() -> None:
    client = _client_with_crashing_route()

    response = client.get(
        "/documents/00000000-0000-0000-0000-000000000000",
        headers={"Origin": _FRONTEND_ORIGIN},
    )

    assert response.status_code == 404
    assert "detail" in response.json()
    assert response.headers.get("access-control-allow-origin") == _FRONTEND_ORIGIN
