"""Celery application factory for all services."""

from celery import Celery
from .config import get_settings


def create_celery_app(
    service_name: str,
    broker_url: str | None = None,
    result_backend: str | None = None,
    include: list[str] | None = None,
) -> Celery:
    """Create a Celery app with standard configuration.

    Args:
        service_name: Name of the service (e.g., "transcription", "translation")
        broker_url: Redis broker URL (defaults to settings)
        result_backend: Redis result backend URL (defaults to settings)
        include: List of modules to include for task discovery

    Returns:
        Configured Celery application
    """
    settings = get_settings()

    broker = broker_url or settings.redis_url
    backend = result_backend or settings.redis_url

    include_modules = include or [f"services.{service_name}.worker"]

    app = Celery(
        service_name,
        broker=broker,
        backend=backend,
        include=include_modules,
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_time_limit=3600,
        task_soft_time_limit=3300,
        broker_connection_retry_on_startup=True,
        broker_connection_max_retries=10,
        broker_connection_retry_interval=5,
        result_expires=86400,
    )

    return app
