"""OCR service - Celery worker tasks."""

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging

logger = setup_logging("ocr-worker")

celery_app = create_celery_app("ocr")


@celery_app.task(bind=True, name="ocr.process")
def ocr_task(self, file_path: str, language: str = "en") -> dict:
    """Process image with OCR."""
    logger.info(f"Starting OCR: {file_path}")
    # TODO: Move OCR logic from current app.py
    return {"text": "OCR not yet implemented", "confidence": None, "boxes": []}
