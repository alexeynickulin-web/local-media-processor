"""OCR service - Celery worker tasks."""

import os
import torch
import logging

import easyocr

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging
from shared.exceptions import ProcessingError

logger = setup_logging("ocr-worker")

celery_app = create_celery_app("ocr")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")

OCR_LANGUAGES = {
    "en": ["en"],
    "ru": ["ru"],
    "en_ru": ["en", "ru"],
    "fr": ["fr"],
    "de": ["de"],
    "es": ["es"],
    "zh": ["ch_sim", "en"],
    "ja": ["ja", "en"],
}


def get_ocr_reader(language: str):
    """Load or get cached EasyOCR reader."""
    langs = OCR_LANGUAGES.get(language, ["en"])
    key = "_".join(sorted(langs))

    return easyocr.Reader(
        langs,
        gpu=DEVICE == "cuda",
        model_storage_directory=os.path.join(MODELS_DIR, "ocr"),
    )


@celery_app.task(bind=True, name="ocr.extract_text")
def ocr_task(
    self,
    file_path: str,
    language: str = "en",
) -> dict:
    """Extract text from image.

    Args:
        file_path: Path to image file
        language: Language code for OCR

    Returns:
        Dict with OCR result
    """
    logger.info(f"Extracting OCR from: {file_path}")

    try:
        if not os.path.exists(file_path):
            raise ProcessingError(f"File not found: {file_path}")

        reader = get_ocr_reader(language)

        result = reader.readtext(file_path, detail=0, paragraph=True)

        extracted_text = " ".join(result)

        return {
            "text": extracted_text,
            "language": language,
            "file_path": file_path,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise ProcessingError(f"OCR failed: {e}")