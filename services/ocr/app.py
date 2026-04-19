"""OCR service - FastAPI application."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from datetime import datetime

from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging
from shared.shared.models import TaskResponse, HealthResponse, TaskStatus

logger = setup_logging("ocr")

celery_app = create_celery_app("ocr")

app = FastAPI(
    title="OCR Service",
    description="Image OCR using EasyOCR",
    version="0.1.0",
)


class OCRRequest(BaseModel):
    file_path: str = Field(..., description="Path to image file")
    language: str = Field(default="en", description="Language for OCR")


def get_task_result(task_id: str) -> dict | None:
    """Get Celery task result."""
    result = celery_app.AsyncResult(task_id)
    if result.ready():
        return result.result
    return None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        service="ocr",
        status="healthy",
        timestamp=datetime.utcnow(),
    )


@app.post("/ocr", response_model=TaskResponse)
async def submit_ocr(request: OCRRequest):
    from .worker import ocr_task

    logger.info(f"OCR requested: {request.file_path}")

    task = ocr_task.delay(request.file_path, request.language)

    return TaskResponse(
        task_id=task.id,
        status=TaskStatus.PENDING,
        message=f"OCR task submitted for {request.file_path}",
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