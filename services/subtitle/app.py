"""Subtitle service - FastAPI application."""

from fastapi import FastAPI
from shared.models import SubtitleRequest, TaskResponse, HealthResponse
from shared.logging_config import setup_logging
from datetime import datetime

logger = setup_logging("subtitle")

app = FastAPI(title="Subtitle Service", description="Subtitle generation using pysrt", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(service="subtitle", status="healthy", timestamp=datetime.utcnow())


@app.post("/subtitle", response_model=TaskResponse)
async def submit_subtitle(request: SubtitleRequest):
    logger.info(f"Subtitle generation requested")
    return TaskResponse(task_id="placeholder", status="pending", message="Subtitle service not yet fully implemented")
