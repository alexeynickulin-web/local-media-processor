"""Transcription service - Celery worker tasks."""

import os
import tempfile
import torch
import logging

import ffmpeg
import whisperx
from faster_whisper import WhisperModel

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging
from shared.exceptions import ProcessingError

logger = setup_logging("transcription-worker")

celery_app = create_celery_app("transcription")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")


def extract_audio_ffmpeg(video_path: str) -> str:
    """Extract audio from video using ffmpeg."""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tf.close()
    output_path = tf.name

    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec='pcm_s16le', ac=1, ar=16000, vn=None, loglevel="error")
            .run(overwrite_output=True)
        )
        return output_path
    except Exception as e:
        raise ProcessingError(f"Failed to extract audio: {e}")


def get_whisper_model(model_name: str):
    """Load or get cached whisper model."""
    model_dir = os.path.join(MODELS_DIR, "whisper")
    return WhisperModel(
        model_name,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root=model_dir,
    )


@celery_app.task(bind=True, name="transcription.transcribe")
def transcribe_task(
    self,
    file_path: str,
    model_name: str = "medium",
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
    logger.info(f"Starting transcription: {file_path} with model {model_name}")

    try:
        audio_path = extract_audio_ffmpeg(file_path)

        try:
            model = get_whisper_model(model_name)

            result = model.transcribe(
                audio_path,
                batch_size=16,
                language=language,
            )

            detected_lang = result.get("language", language or "en")
            segments = result.get("segments", [])

            full_text = " ".join([s["text"].strip() for s in segments])

            try:
                align_model, metadata = whisperx.load_align_model(
                    language_code=detected_lang,
                    device=DEVICE,
                )
                aligned_result = whisperx.align(
                    segments,
                    align_model,
                    metadata,
                    audio_path,
                    DEVICE,
                    return_char_alignments=False,
                )
                segments = aligned_result["segments"]
                full_text = " ".join([s["text"].strip() for s in segments])

                del align_model, metadata
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Alignment failed: {e}, using raw segments")

            os.unlink(audio_path)

            return {
                "text": full_text,
                "segments": segments,
                "language": detected_lang,
                "model": model_name,
                "status": "completed",
            }

        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise ProcessingError(f"Transcription failed: {e}")