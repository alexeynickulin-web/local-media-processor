"""Subtitle service - Celery worker tasks."""

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging

logger = setup_logging("subtitle-worker")

celery_app = create_celery_app("subtitle")


@celery_app.task(bind=True, name="subtitle.generate")
def subtitle_task(self, transcription_result: dict, format: str = "srt") -> dict:
    """Generate subtitles from transcription result."""
    logger.info(f"Starting subtitle generation: {format}")
    # TODO: Move subtitle logic from current app.py
    return {"subtitle_path": "", "format": format}
