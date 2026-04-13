"""Transcription service - FastAPI application."""

from fastapi import FastAPI, HTTPException
from shared.models import (
    TranscriptionRequest,
    TranscriptionResult,
    TaskResponse,
    HealthResponse,
)
from shared.logging_config import setup_logging
from shared.exceptions import ProcessingError
from datetime import datetime

logger = setup_logging("transcription")

app = FastAPI(
    title="Transcription Service",
    description="Audio/video transcription using faster-whisper",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        service="transcription",
        status="healthy",
        timestamp=datetime.utcnow(),
    )


@app.post("/transcribe", response_model=TaskResponse)
async def submit_transcription(request: TranscriptionRequest):
    """Submit a transcription task.

    The task will be processed asynchronously via Celery.
    Returns a task_id that can be used to check status.
    """
    # TODO: Implement actual Celery task submission
    # from .worker import transcribe_task
    # task = transcribe_task.delay(request.file_path, request.model_name, request.language)

    logger.info(f"Transcription requested: {request.file_path}")

    return TaskResponse(
        task_id="placeholder",
        status="pending",
        message="Transcription service not yet fully implemented",
    )


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a transcription task."""
    # TODO: Implement actual task status lookup
    raise HTTPException(status_code=404, detail="Not yet implemented")
