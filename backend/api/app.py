"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.exception_handlers import register_exception_handlers
from backend.api.middleware import UnhandledErrorResponseMiddleware
from backend.api.routes.documents import router as documents_router
from backend.api.routes.embedding import router as embedding_router
from backend.api.routes.evaluation import router as evaluation_router
from backend.api.routes.generation import router as generation_router
from backend.api.routes.graph import router as graph_router
from backend.api.routes.health import router as health_router
from backend.api.routes.indexing import router as indexing_router
from backend.api.routes.parsing import router as parsing_router
from backend.api.routes.representation import router as representation_router
from backend.config.settings import get_settings
from backend.shared.logging import configure_logging


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance.

    Configuration and logging are resolved here rather than at import time,
    so that every application instance (including ones built by tests) is
    fully and independently configured.

    Returns:
        A configured :class:`~fastapi.FastAPI` application instance.
    """
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.environment == "development",
    )

    # Added before CORSMiddleware so CORS ends up outermost: unhandled
    # errors become a JSON 500 here, then pick up CORS headers on the way
    # out -- see backend/api/middleware.py for why this ordering matters.
    app.add_middleware(UnhandledErrorResponseMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(parsing_router)
    app.include_router(representation_router)
    app.include_router(embedding_router)
    app.include_router(indexing_router)
    app.include_router(graph_router)
    app.include_router(generation_router)
    app.include_router(evaluation_router)
    register_exception_handlers(app)

    return app
