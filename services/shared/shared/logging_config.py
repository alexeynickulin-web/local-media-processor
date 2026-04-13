"""Logging configuration for all microservices."""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from .config import get_settings


def setup_logging(
    service_name: str,
    log_level: str | None = None,
    log_file: str | None = None,
) -> logging.Logger:
    """Configure logging for a service.

    Args:
        service_name: Name of the service (used in logger name)
        log_level: Logging level (defaults to settings)
        log_file: Path to log file (optional)

    Returns:
        Configured logger instance
    """
    settings = get_settings()
    level = log_level or settings.log_level

    logger = logging.getLogger(f"media-processor.{service_name}")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
