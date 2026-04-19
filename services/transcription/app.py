"""Transcription service - FastAPI application."""

import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from datetime import datetime
from typing import Any

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging
from shared.shared.models import TaskResponse, HealthResponse, TaskStatus

logger = setup_logging("transcription")

celery_app = create_celery_app("transcription")

app = FastAPI(
    title="Transcription Service",
    description="Audio/video transcription using faster-whisper",
    version="0.1.0",
)

whisper_models = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]


class TranscriptionRequest(BaseModel):
    file_path: str = Field(..., description="Path to audio/video file")
    model_name: str = Field(default="medium", description="Whisper model name")
    language: str | None = Field(default=None, description="Source language code")


class TranscriptionResult(BaseModel):
    text: str
    segments: list[dict[str, Any]] | None = None
    language: str | None = None
    model: str


def get_task_result(task_id: str) -> dict | None:
    """Get Celery task result."""
    result = celery_app.AsyncResult(task_id)
    if result.ready():
        return result.result
    return None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        service="transcription",
        status="healthy",
        timestamp=datetime.utcnow(),
    )


@app.post("/transcribe", response_model=TaskResponse)
async def submit_transcription(request: TranscriptionRequest, background_tasks: BackgroundTasks):
    from .worker import transcribe_task

    logger.info(f"Transcription requested: {request.file_path}")

    task = transcribe_task.delay(
        request.file_path,
        request.model_name,
        request.language,
    )

    return TaskResponse(
        task_id=task.id,
        status=TaskStatus.PENDING,
        message=f"Transcription task submitted for {request.file_path}",
    )


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    result = get_task_result(task_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Task not found or still processing")

    return {
        "task_id": task_id,
        "status": "completed",
        "result": result,
    }


@app.get("/result/{task_id}")
async def get_task_result_endpoint(task_id: str):
    result = get_task_result(task_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Task not found or still processing")

    return result