"""Translation service - FastAPI application."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from datetime import datetime
from typing import Any

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging
from shared.shared.models import TaskResponse, HealthResponse, TaskStatus

logger = setup_logging("translation")

celery_app = create_celery_app("translation")

app = FastAPI(
    title="Translation Service",
    description="Text translation using NLLB models",
    version="0.1.0",
)


class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    model_name: str = Field(default="facebook/nllb-200-distilled-600M")


class SegmentsTranslationRequest(BaseModel):
    segments: list[dict[str, Any]] = Field(..., description="Transcription segments")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    model_name: str = Field(default="facebook/nllb-200-distilled-600M")


def get_task_result(task_id: str) -> dict | None:
    """Get Celery task result."""
    result = celery_app.AsyncResult(task_id)
    if result.ready():
        return result.result
    return None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        service="translation",
        status="healthy",
        timestamp=datetime.utcnow(),
    )


@app.post("/translate", response_model=TaskResponse)
async def submit_translation(request: TranslationRequest):
    from .worker import translate_task

    logger.info(f"Translation requested: {request.source_language} -> {request.target_language}")

    task = translate_task.delay(
        request.text,
        request.source_language,
        request.target_language,
        request.model_name,
    )

    return TaskResponse(
        task_id=task.id,
        status=TaskStatus.PENDING,
        message=f"Translation task submitted",
    )


@app.post("/translate-segments", response_model=TaskResponse)
async def submit_segments_translation(request: SegmentsTranslationRequest):
    from .worker import translate_segments_task

    logger.info(f"Segment translation: {len(request.segments)} segments")

    task = translate_segments_task.delay(
        request.segments,
        request.source_language,
        request.target_language,
        request.model_name,
    )

    return TaskResponse(
        task_id=task.id,
        status=TaskStatus.PENDING,
        message=f"Segment translation task submitted",
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