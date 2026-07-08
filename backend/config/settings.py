"""Centralized application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the backend application.

    Values are loaded from environment variables prefixed with ``KIMRAG_``,
    optionally via a local ``.env`` file. Instances should be obtained
    through :func:`get_settings` rather than constructed directly, so that
    configuration is read once per process.

    Attributes:
        app_name: Human-readable application name, used in API metadata.
        app_version: Application version, used in API metadata.
        environment: Deployment environment the process is running in.
        host: Interface the HTTP server binds to.
        port: Port the HTTP server binds to.
        reload: Whether the server should reload on code changes.
        log_level: Minimum severity of log records emitted by the application.
        storage_root: Base directory document workspaces are created under.
        max_upload_size_bytes: Maximum permitted size for an uploaded document.
        parsed_storage_root: Base directory parsed document artifacts are
            written under.
        docling_ocr_enabled: Whether Docling runs OCR for scanned/image
            -based pages. Disabled by default: most scientific papers are
            born-digital with a real text layer, and enabling this pulls in
            an extra OCR model download in fresh environments.
    """

    app_name: str = "Knowledge-Infused Multimodal RAG"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    storage_root: Path = Path("data/raw")
    max_upload_size_bytes: int = 50 * 1024 * 1024
    parsed_storage_root: Path = Path("data/parsed")
    docling_ocr_enabled: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KIMRAG_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide, cached application settings.

    Returns:
        The single :class:`Settings` instance for this process.
    """
    return Settings()
