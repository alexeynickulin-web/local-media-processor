"""Shared configuration for all microservices."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class SharedSettings(BaseSettings):
    """Base settings shared across all services."""

    # Redis configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    # Service discovery
    api_gateway_url: str = "http://localhost:8000"
    transcription_service_url: str = "http://localhost:8001"
    translation_service_url: str = "http://localhost:8002"
    ocr_service_url: str = "http://localhost:8003"
    tts_service_url: str = "http://localhost:8004"
    subtitle_service_url: str = "http://localhost:8005"

    # File paths
    models_dir: str = "./models"
    temp_dir: str = "/tmp/media-processor"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # or "text"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> SharedSettings:
    """Get cached settings instance."""
    return SharedSettings()
