"""Subtitle service - FastAPI application."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any

from datetime import datetime

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging
from shared.shared.models import TaskResponse, HealthResponse, TaskStatus

logger = setup_logging("subtitle")

celery_app = create_celery_app("subtitle")

app = FastAPI(
    title="Subtitle Service",
    description="Subtitle generation from transcription segments",
    version="0.1.0",
)


class SubtitleRequest(BaseModel):
    segments: list[dict[str, Any]] = Field(..., description="Transcription segments")
    format: str = Field(default="srt", description="Subtitle format (srt, vtt)")
    base_name: str = Field(default="output", description="Base name for output file")


SUPPORTED_FORMATS = ["srt", "vtt"]


def get_task_result(task_id: str) -> dict | None:
    """Get Celery task result."""
    result = celery_app.AsyncResult(task_id)
    if result.ready():
        return result.result
    return None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        service="subtitle",
        status="healthy",
        timestamp=datetime.utcnow(),
    )


@app.post("/subtitle", response_model=TaskResponse)
async def submit_subtitle(request: SubtitleRequest):
    from .worker import generate_subtitle_task

    if request.format not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")

    logger.info(f"Subtitle requested: {len(request.segments)} segments, format={request.format}")

    task = generate_subtitle_task.delay(
        request.segments,
        request.format,
        request.base_name,
    )

    return TaskResponse(
        task_id=task.id,
        status=TaskStatus.PENDING,
        message=f"Subtitle task submitted",
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