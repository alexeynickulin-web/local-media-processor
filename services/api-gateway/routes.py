"""API Gateway routes."""

from fastapi import APIRouter, HTTPException, Depends
from shared.models import (
    TaskResponse,
    TaskResult,
    TranscriptionRequest,
    TranslationRequest,
    OCRRequest,
    TTSRequest,
    SubtitleRequest,
    BatchRequest,
    BatchStatus,
    HealthResponse,
)
from shared.exceptions import MediaProcessorError
from datetime import datetime

router = APIRouter()


# TODO: Implement route proxies to backend services
# These routes will forward requests to appropriate microservices


@router.post("/transcribe", response_model=TaskResponse)
async def transcribe(request: TranscriptionRequest):
    """Submit transcription task.

    Forwards request to transcription service via Celery.
    """
    # TODO: Implement actual Celery task submission
    return TaskResponse(
        task_id="placeholder",
        status="pending",
        message="Transcription task submitted (not yet implemented)",
    )


@router.post("/translate", response_model=TaskResponse)
async def translate(request: TranslationRequest):
    """Submit translation task.

    Forwards request to translation service via Celery.
    """
    # TODO: Implement actual Celery task submission
    return TaskResponse(
        task_id="placeholder",
        status="pending",
        message="Translation task submitted (not yet implemented)",
    )


@router.post("/ocr", response_model=TaskResponse)
async def ocr(request: OCRRequest):
    """Submit OCR task.

    Forwards request to OCR service via Celery.
    """
    # TODO: Implement actual Celery task submission
    return TaskResponse(
        task_id="placeholder",
        status="pending",
        message="OCR task submitted (not yet implemented)",
    )


@router.post("/tts", response_model=TaskResponse)
async def tts(request: TTSRequest):
    """Submit TTS task.

    Forwards request to TTS service via Celery.
    """
    # TODO: Implement actual Celery task submission
    return TaskResponse(
        task_id="placeholder",
        status="pending",
        message="TTS task submitted (not yet implemented)",
    )


@router.post("/subtitle", response_model=TaskResponse)
async def subtitle(request: SubtitleRequest):
    """Submit subtitle generation task.

    Forwards request to subtitle service via Celery.
    """
    # TODO: Implement actual Celery task submission
    return TaskResponse(
        task_id="placeholder",
        status="pending",
        message="Subtitle task submitted (not yet implemented)",
    )


@router.post("/batch", response_model=TaskResponse)
async def batch_process(request: BatchRequest):
    """Submit batch processing task."""
    # TODO: Implement actual Celery task submission
    return TaskResponse(
        task_id="placeholder",
        status="pending",
        message="Batch task submitted (not yet implemented)",
    )


@router.get("/status/{task_id}", response_model=TaskResult)
async def get_task_status(task_id: str):
    """Get status of a task.

    Queries Redis for task result.
    """
    # TODO: Implement actual task status lookup
    return TaskResult(
        task_id=task_id,
        status="pending",
        error="Task status not yet implemented",
    )


@router.get("/batch/{batch_id}", response_model=BatchStatus)
async def get_batch_status(batch_id: str):
    """Get status of a batch job."""
    # TODO: Implement actual batch status lookup
    raise HTTPException(status_code=404, detail="Not yet implemented")
