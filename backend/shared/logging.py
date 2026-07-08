"""Centralized logging configuration for the application."""

import logging.config

from backend.config.settings import Settings


def configure_logging(settings: Settings) -> None:
    """Configure root logging handlers and formatting for the process.

    Logs are written to stdout only, in line with container-friendly
    (12-factor) logging practices — log routing and persistence are left to
    the deployment environment rather than handled by the application.

    Args:
        settings: Application settings providing the desired log level.
    """
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": settings.log_level,
        },
    }
    logging.config.dictConfig(logging_config)
