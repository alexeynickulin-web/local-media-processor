"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Possible task states."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskRequest(BaseModel):
    """Base model for task requests."""

    pass


class TaskResponse(BaseModel):
    """Base model for task responses."""

    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    message: str | None = None


class TaskResult(BaseModel):
    """Model for completed task results."""

    task_id: str
    status: TaskStatus
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


class HealthResponse(BaseModel):
    """Model for health check responses."""

    service: str
    status: str
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Transcription models
class TranscriptionRequest(TaskRequest):
    """Request model for transcription."""

    file_path: str = Field(..., description="Path to audio/video file")
    model_name: str = Field(default="base", description="Whisper model name")
    language: str | None = Field(default=None, description="Source language code")


class TranscriptionResult(BaseModel):
    """Result model for transcription."""

    text: str
    segments: list[dict[str, Any]] | None = None
    language: str | None = None
    duration: float | None = None


# Translation models
class TranslationRequest(TaskRequest):
    """Request model for translation."""

    text: str = Field(..., description="Text to translate")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    model_name: str = Field(default="nllb-200-distilled-600M", description="Translation model")


class TranslationResult(BaseModel):
    """Result model for translation."""

    translated_text: str
    source_language: str
    target_language: str


# OCR models
class OCRRequest(TaskRequest):
    """Request model for OCR."""

    file_path: str = Field(..., description="Path to image file")
    language: str = Field(default="en", description="Language for OCR")


class OCRResult(BaseModel):
    """Result model for OCR."""

    text: str
    confidence: float | None = None
    boxes: list[dict[str, Any]] | None = None


# TTS models
class TTSRequest(TaskRequest):
    """Request model for text-to-speech."""

    text: str = Field(..., description="Text to convert to speech")
    language: str = Field(..., description="Language code")
    voice: str | None = Field(default=None, description="Voice name")


class TTSResult(BaseModel):
    """Result model for TTS."""

    audio_path: str
    duration: float | None = None


# Subtitle models
class SubtitleRequest(TaskRequest):
    """Request model for subtitle generation."""

    transcription_result: TranscriptionResult = Field(
        ..., description="Transcription result"
    )
    format: str = Field(default="srt", description="Subtitle format (srt, vtt, etc.)")


class SubtitleResult(BaseModel):
    """Result model for subtitle generation."""

    subtitle_path: str
    format: str


# Batch processing models
class BatchRequest(TaskRequest):
    """Request model for batch processing."""

    file_paths: list[str] = Field(..., description="List of file paths to process")
    task_type: str = Field(..., description="Type of batch task")
    options: dict[str, Any] = Field(default_factory=dict, description="Task options")


class BatchStatus(BaseModel):
    """Model for batch processing status."""

    batch_id: str
    total: int
    completed: int
    failed: int
    status: TaskStatus
    results: list[TaskResult] = Field(default_factory=list)
