"""Shared components for all microservices."""

__version__ = "0.1.0"

from .config import get_settings, SharedSettings
from .celery_app import create_celery_app
from .models import (
    TaskStatus,
    TaskRequest,
    TaskResponse,
    TaskResult,
    HealthResponse,
    TranscriptionRequest,
    TranscriptionResult,
    TranslationRequest,
    TranslationResult,
    OCRRequest,
    OCRResult,
    TTSRequest,
    TTSResult,
    SubtitleRequest,
    SubtitleResult,
    BatchRequest,
    BatchStatus,
)
from .exceptions import (
    MediaProcessorError,
    FileNotFoundError,
    InvalidFileError,
    ModelLoadError,
    ProcessingError,
    TimeoutError,
    ValidationError,
)
from .file_io import (
    validate_file_exists,
    validate_file_extension,
    create_temp_file,
    safe_remove_file,
    ensure_directory,
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
)
from .language import (
    validate_language_code,
    get_nllb_code,
    get_language_name,
    SUPPORTED_LANGUAGES,
    NLLB_LANGUAGE_CODES,
)
from .logging_config import setup_logging
from .redis_client import get_redis_client, test_redis_connection

__all__ = [
    "get_settings",
    "SharedSettings",
    "create_celery_app",
    "TaskStatus",
    "TaskRequest",
    "TaskResponse",
    "TaskResult",
    "HealthResponse",
    "TranscriptionRequest",
    "TranscriptionResult",
    "TranslationRequest",
    "TranslationResult",
    "OCRRequest",
    "OCRResult",
    "TTSRequest",
    "TTSResult",
    "SubtitleRequest",
    "SubtitleResult",
    "BatchRequest",
    "BatchStatus",
    "MediaProcessorError",
    "FileNotFoundError",
    "InvalidFileError",
    "ModelLoadError",
    "ProcessingError",
    "TimeoutError",
    "ValidationError",
    "validate_file_exists",
    "validate_file_extension",
    "create_temp_file",
    "safe_remove_file",
    "ensure_directory",
    "ALLOWED_AUDIO_EXTENSIONS",
    "ALLOWED_VIDEO_EXTENSIONS",
    "ALLOWED_IMAGE_EXTENSIONS",
    "validate_language_code",
    "get_nllb_code",
    "get_language_name",
    "SUPPORTED_LANGUAGES",
    "NLLB_LANGUAGE_CODES",
    "setup_logging",
    "get_redis_client",
    "test_redis_connection",
]
