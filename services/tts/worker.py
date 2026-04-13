"""TTS service - Celery worker tasks."""

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging

logger = setup_logging("tts-worker")

celery_app = create_celery_app("tts")


@celery_app.task(bind=True, name="tts.synthesize")
def tts_task(self, text: str, language: str, voice: str | None = None) -> dict:
    """Synthesize speech from text."""
    logger.info(f"Starting TTS: {language}")
    # TODO: Move TTS logic from current app.py
    return {"audio_path": "", "duration": 0.0}
