"""Translation service - FastAPI application."""

from fastapi import FastAPI, HTTPException
from shared.models import TranslationRequest, TaskResponse, HealthResponse
from shared.logging_config import setup_logging
from datetime import datetime

logger = setup_logging("translation")

app = FastAPI(
    title="Translation Service",
    description="Text translation using NLLB models",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        service="translation",
        status="healthy",
        timestamp=datetime.utcnow(),
    )


@app.post("/translate", response_model=TaskResponse)
async def submit_translation(request: TranslationRequest):
    """Submit a translation task."""
    logger.info(f"Translation requested: {request.source_language} -> {request.target_language}")

    return TaskResponse(
        task_id="placeholder",
        status="pending",
        message="Translation service not yet fully implemented",
    )
