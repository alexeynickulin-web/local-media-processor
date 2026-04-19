"""TTS service - Celery worker tasks."""

import os
import asyncio
import logging
import tempfile

import edge_tts

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging
from shared.exceptions import ProcessingError

logger = setup_logging("tts-worker")

celery_app = create_celery_app("tts")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/tts")

VOICE_MAP = {
    "en": "en-US-JennyNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KlaudiaNeural",
    "es": "es-ES-ElviraNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural",
    "it": "it-IT-ElsaNeural",
}


def get_voice_for_language(language: str) -> str:
    """Get default voice for language."""
    return VOICE_MAP.get(language, "en-US-JennyNeural")


async def _synthesize_async(text: str, voice: str, output_path: str) -> dict:
    """Async helper for edge-tts."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

    file_size = os.path.getsize(output_path)
    duration = file_size / 16000

    return {
        "audio_path": output_path,
        "voice": voice,
        "text_length": len(text),
        "status": "completed",
    }


@celery_app.task(bind=True, name="tts.synthesize")
def synthesize_task(
    self,
    text: str,
    language: str,
    voice: str | None = None,
) -> dict:
    """Synthesize text to speech.

    Args:
        text: Text to convert to speech
        language: Language code
        voice: Voice name (optional, auto-select if None)

    Returns:
        Dict with TTS result
    """
    logger.info(f"Synthesizing TTS for language: {language}")

    try:
        if not text:
            raise ProcessingError("Text is empty")

        selected_voice = voice or get_voice_for_language(language)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=OUTPUT_DIR)
        tf.close()
        output_path = tf.name

        result = asyncio.run(_synthesize_async(text, selected_voice, output_path))

        return {
            **result,
            "language": language,
            "text": text[:100],
        }

    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise ProcessingError(f"TTS failed: {e}")