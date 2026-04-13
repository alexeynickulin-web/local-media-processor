"""Transcription service - Celery worker tasks."""

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging
from shared.exceptions import ProcessingError, FileNotFoundError

logger = setup_logging("transcription-worker")

# Create Celery app
celery_app = create_celery_app("transcription")


@celery_app.task(bind=True, name="transcription.transcribe")
def transcribe_task(
    self,
    file_path: str,
    model_name: str = "base",
    language: str | None = None,
) -> dict:
    """Transcribe audio/video file.

    Args:
        file_path: Path to the file to transcribe
        model_name: Whisper model name to use
        language: Source language code (optional, auto-detect if None)

    Returns:
        Dict with transcription result
    """
    # TODO: Move transcription logic from current app.py
    # TODO: Implement actual transcription logic
    logger.info(f"Starting transcription: {file_path} with model {model_name}")

    # Placeholder implementation
    return {
        "text": "Transcription not yet implemented",
        "segments": [],
        "language": language,
        "duration": 0.0,
    }
