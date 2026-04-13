"""Translation service - Celery worker tasks."""

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging
from shared.language import get_nllb_code

logger = setup_logging("translation-worker")

celery_app = create_celery_app("translation")


@celery_app.task(bind=True, name="translation.translate")
def translate_task(
    self,
    text: str,
    source_language: str,
    target_language: str,
    model_name: str = "nllb-200-distilled-600M",
) -> dict:
    """Translate text.

    Args:
        text: Text to translate
        source_language: Source language code
        target_language: Target language code
        model_name: Translation model to use

    Returns:
        Dict with translation result
    """
    logger.info(f"Starting translation: {source_language} -> {target_language}")

    # TODO: Move translation logic from current app.py
    # TODO: Implement chunked translation for long texts

    return {
        "translated_text": "Translation not yet implemented",
        "source_language": source_language,
        "target_language": target_language,
    }
