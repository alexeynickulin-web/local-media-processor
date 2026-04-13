"""API Gateway routes with full service integration."""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from shared.models import (
    TaskResponse,
    TaskResult,
    TaskStatus,
    TranscriptionRequest,
    TranslationRequest,
    OCRRequest,
    TTSRequest,
    SubtitleRequest,
    BatchRequest,
    BatchStatus,
    HealthResponse,
)
from shared.exceptions import MediaProcessorError, ProcessingError
from .task_manager import task_manager
from .service_client import router as service_router
from .auth import verify_api_key
from datetime import datetime
from typing import Any

router = APIRouter()


@router.post("/transcribe", response_model=TaskResponse)
async def transcribe(
    request: TranscriptionRequest,
    auth: str | None = Depends(verify_api_key),
):
    """Submit transcription task.

    Forwards request to transcription service via Celery.
    """
    try:
        task_id = task_manager.submit_task(
            "transcription.transcribe",
            kwargs={
                "file_path": request.file_path,
                "model_name": request.model_name,
                "language": request.language,
            },
            queue="transcription",
        )

        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Transcription task submitted",
        )

    except MediaProcessorError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/translate", response_model=TaskResponse)
async def translate(
    request: TranslationRequest,
    auth: str | None = Depends(verify_api_key),
):
    """Submit translation task.

    Forwards request to translation service via Celery.
    """
    try:
        task_id = task_manager.submit_task(
            "translation.translate",
            kwargs={
                "text": request.text,
                "source_language": request.source_language,
                "target_language": request.target_language,
                "model_name": request.model_name,
            },
            queue="translation",
        )

        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Translation task submitted",
        )

    except MediaProcessorError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr", response_model=TaskResponse)
async def ocr(
    request: OCRRequest,
    auth: str | None = Depends(verify_api_key),
):
    """Submit OCR task.

    Forwards request to OCR service via Celery.
    """
    try:
        task_id = task_manager.submit_task(
            "ocr.process",
            kwargs={
                "file_path": request.file_path,
                "language": request.language,
            },
            queue="ocr",
        )

        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="OCR task submitted",
        )

    except MediaProcessorError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tts", response_model=TaskResponse)
async def tts(
    request: TTSRequest,
    auth: str | None = Depends(verify_api_key),
):
    """Submit TTS task.

    Forwards request to TTS service via Celery.
    """
    try:
        task_id = task_manager.submit_task(
            "tts.synthesize",
            kwargs={
                "text": request.text,
                "language": request.language,
                "voice": request.voice,
            },
            queue="tts",
        )

        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="TTS task submitted",
        )

    except MediaProcessorError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subtitle", response_model=TaskResponse)
async def subtitle(
    request: SubtitleRequest,
    auth: str | None = Depends(verify_api_key),
):
    """Submit subtitle generation task.

    Forwards request to subtitle service via Celery.
    """
    try:
        task_id = task_manager.submit_task(
            "subtitle.generate",
            kwargs={
                "transcription_result": request.transcription_result.model_dump(),
                "format": request.format,
            },
            queue="subtitle",
        )

        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Subtitle task submitted",
        )

    except MediaProcessorError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=TaskResponse)
async def batch_process(
    request: BatchRequest,
    auth: str | None = Depends(verify_api_key),
):
    """Submit batch processing task.

    Submits multiple tasks and returns a batch ID.
    """
    try:
        # Submit individual tasks
        task_ids = []
        for file_path in request.file_paths:
            task_id = task_manager.submit_task(
                f"{request.task_type}.process",
                kwargs={"file_path": file_path, **request.options},
                queue=request.task_type,
            )
            task_ids.append(task_id)

        return TaskResponse(
            task_id=f"batch_{datetime.utcnow().timestamp()}",
            status=TaskStatus.PENDING,
            message=f"Batch submitted with {len(task_ids)} tasks",
        )

    except MediaProcessorError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=TaskResult)
async def get_task_status(
    task_id: str,
    auth: str | None = Depends(verify_api_key),
):
    """Get status of a task.

    Queries Celery for task result.
    """
    try:
        status_data = task_manager.get_task_status(task_id)

        return TaskResult(
            task_id=task_id,
            status=TaskStatus(status_data["status"]),
            result=status_data.get("result"),
            error=status_data.get("error"),
            completed_at=status_data.get("completed_at"),
        )

    except MediaProcessorError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")


@router.get("/batch/{batch_id}", response_model=BatchStatus)
async def get_batch_status(
    batch_id: str,
    auth: str | None = Depends(verify_api_key),
):
    """Get status of a batch job."""
    # TODO: Implement batch status tracking
    raise HTTPException(status_code=404, detail="Batch status not yet implemented")


@router.post("/cancel/{task_id}")
async def cancel_task(
    task_id: str,
    auth: str | None = Depends(verify_api_key),
):
    """Cancel a running task."""
    try:
        success = task_manager.revoke_task(task_id, terminate=True)
        if success:
            return {"message": f"Task {task_id} cancelled"}
        raise HTTPException(status_code=500, detail="Failed to cancel task")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/health")
async def services_health():
    """Check health of all backend services."""
    try:
        health = await service_router.check_all_services()
        return {
            "api_gateway": "healthy",
            "services": health,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
