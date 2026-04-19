"""Subtitle service - Celery worker tasks."""

import os
import logging
import tempfile

import pysrt

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging
from shared.exceptions import ProcessingError

logger = setup_logging("subtitle-worker")

celery_app = create_celery_app("subtitle")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/subtitles")


@celery_app.task(bind=True, name="subtitle.generate")
def generate_subtitle_task(
    self,
    segments: list[dict],
    format: str = "srt",
    base_name: str = "output",
) -> dict:
    """Generate subtitle file from segments.

    Args:
        segments: List of segments with start/end/text
        format: Subtitle format (srt, vtt)
        base_name: Base name for output file

    Returns:
        Dict with subtitle result
    """
    logger.info(f"Generating {format} subtitles for {len(segments)} segments")

    try:
        if not segments:
            raise ProcessingError("No segments provided")

        subs = pysrt.SubRipFile()
        for i, s in enumerate(segments):
            subs.append(
                pysrt.SubRipItem(
                    i + 1,
                    start=pysrt.SubRipTime(seconds=s["start"]),
                    end=pysrt.SubRipTime(seconds=s["end"]),
                    text=s["text"],
                )
            )

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if format == "vtt":
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}.vtt")
            subs.save(output_path, encoding="utf-8")
        else:
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}.srt")
            subs.save(output_path, encoding="utf-8")

        file_size = os.path.getsize(output_path)

        return {
            "subtitle_path": output_path,
            "format": format,
            "segment_count": len(segments),
            "file_size": file_size,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Subtitle generation failed: {e}")
        raise ProcessingError(f"Subtitle generation failed: {e}")