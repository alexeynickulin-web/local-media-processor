"""TTS service - FastAPI application."""

from fastapi import FastAPI
from shared.models import TTSRequest, TaskResponse, HealthResponse
from shared.logging_config import setup_logging
from datetime import datetime

logger = setup_logging("tts")

app = FastAPI(title="TTS Service", description="Text-to-Speech using edge_tts", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(service="tts", status="healthy", timestamp=datetime.utcnow())


@app.post("/tts", response_model=TaskResponse)
async def submit_tts(request: TTSRequest):
    logger.info(f"TTS requested: {request.language}")
    return TaskResponse(task_id="placeholder", status="pending", message="TTS service not yet fully implemented")
