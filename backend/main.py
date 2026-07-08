"""Process entry point for running the API server."""

import uvicorn

from backend.config.settings import get_settings


def run() -> None:
    """Run the FastAPI application with Uvicorn using resolved settings.

    Host, port, reload, and log level are all sourced from application
    settings so that local development and containerized deployments are
    configured purely through environment variables.
    """
    settings = get_settings()
    uvicorn.run(
        "backend.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
