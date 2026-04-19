"""Translation service - Celery worker tasks."""

import os
import torch
import logging

from transformers import pipeline

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging
from shared.exceptions import ProcessingError

logger = setup_logging("translation-worker")

celery_app = create_celery_app("translation")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")

NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "it": "ita_Latn",
}


def get_nllb_model(model_name: str):
    """Load or get cached NLLB model."""
    return pipeline(
        "translation",
        model=model_name,
        device=0 if DEVICE == "cuda" else -1,
        max_length=512,
    )


@celery_app.task(bind=True, name="translation.translate")
def translate_task(
    self,
    text: str,
    source_language: str,
    target_language: str,
    model_name: str = "facebook/nllb-200-distilled-600M",
) -> dict:
    """Translate text.

    Args:
        text: Text to translate
        source_language: Source language code
        target_language: Target language code
        model_name: NLLB model name

    Returns:
        Dict with translation result
    """
    logger.info(f"Translating from {source_language} to {target_language}")

    try:
        translator = get_nllb_model(model_name)

        src_code = NLLB_LANG_MAP.get(source_language, f"{source_language}_Latn")
        tgt_code = NLLB_LANG_MAP.get(target_language, f"{target_language}_Latn")

        max_chunk = 3000
        if len(text) > max_chunk:
            text = text[:max_chunk]

        result = translator(text, src_lang=src_code, tgt_lang=tgt_code)

        translated_text = result[0]["translation_text"]

        return {
            "text": text,
            "translated_text": translated_text,
            "source_language": source_language,
            "target_language": target_language,
            "model": model_name,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise ProcessingError(f"Translation failed: {e}")


@celery_app.task(bind=True, name="translation.translate_segments")
def translate_segments_task(
    self,
    segments: list[dict],
    source_language: str,
    target_language: str,
    model_name: str = "facebook/nllb-200-distilled-600M",
) -> dict:
    """Translate transcription segments.

    Args:
        segments: List of transcription segments with start/end/text
        source_language: Source language code
        target_language: Target language code
        model_name: NLLB model name

    Returns:
        Dict with translated segments
    """
    logger.info(f"Translating {len(segments)} segments from {source_language} to {target_language}")

    try:
        translator = get_nllb_model(model_name)

        src_code = NLLB_LANG_MAP.get(source_language, f"{source_language}_Latn")
        tgt_code = NLLB_LANG_MAP.get(target_language, f"{target_language}_Latn")

        texts = [s["text"] for s in segments]
        batch_result = translator(texts, src_lang=src_code, tgt_lang=tgt_code, batch_size=16)

        translated_segments = []
        for i, r in enumerate(batch_result):
            translated_segments.append({
                "start": segments[i]["start"],
                "end": segments[i]["end"],
                "text": r["translation_text"],
            })

        return {
            "segments": translated_segments,
            "source_language": source_language,
            "target_language": target_language,
            "model": model_name,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Segment translation failed: {e}")
        raise ProcessingError(f"Translation failed: {e}")