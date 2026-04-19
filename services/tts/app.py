"""TTS service - FastAPI application."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from datetime import datetime

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging
from shared.models import TaskResponse, HealthResponse, TaskStatus

logger = setup_logging("tts")

celery_app = create_celery_app("tts")

app = FastAPI(
    title="TTS Service",
    description="Text-to-speech using edge-tts",
    version="0.1.0",
)


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    language: str = Field(..., description="Language code")
    voice: str | None = Field(default=None, description="Voice name")


SUPPORTED_LANGUAGES = ["en", "ru", "fr", "de", "es", "zh", "ja", "it"]


def get_task_result(task_id: str) -> dict | None:
    """Get Celery task result."""
    result = celery_app.AsyncResult(task_id)
    if result.ready():
        return result.result
    return None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        service="tts",
        status="healthy",
        timestamp=datetime.utcnow(),
    )


@app.post("/tts", response_model=TaskResponse)
async def submit_tts(request: TTSRequest):
    from .worker import synthesize_task

    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {request.language}")

    logger.info(f"TTS requested: language={request.language}")

    task = synthesize_task.apply_async(
        args=[request.text, request.language, request.voice],
        queue="tts",
    )

    return TaskResponse(
        task_id=task.id,
        status=TaskStatus.PENDING,
        message=f"TTS task submitted",
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


@app.get("/download/{task_id}")
async def download_audio(task_id: str):
    from fastapi.responses import FileResponse
    import os

    result = get_task_result(task_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Task not found or still processing")

    audio_path = result.get("audio_path", "")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=os.path.basename(audio_path),
    )