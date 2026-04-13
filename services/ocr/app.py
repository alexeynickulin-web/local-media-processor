"""OCR service - FastAPI application."""

from fastapi import FastAPI
from shared.models import OCRRequest, TaskResponse, HealthResponse
from shared.logging_config import setup_logging
from datetime import datetime

logger = setup_logging("ocr")

app = FastAPI(title="OCR Service", description="Image OCR using EasyOCR", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(service="ocr", status="healthy", timestamp=datetime.utcnow())


@app.post("/ocr", response_model=TaskResponse)
async def submit_ocr(request: OCRRequest):
    logger.info(f"OCR requested: {request.file_path}")
    return TaskResponse(task_id="placeholder", status="pending", message="OCR service not yet fully implemented")
