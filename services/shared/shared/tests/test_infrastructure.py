"""Test Redis and Celery infrastructure."""

import pytest
from shared.config import get_settings
from shared.celery_app import create_celery_app
from shared.redis_client import get_redis_client, test_redis_connection


class TestSettings:
    """Test shared settings."""

    def test_settings_load(self):
        """Test that settings load without errors."""
        settings = get_settings()
        assert settings.redis_host is not None
        assert settings.redis_port > 0

    def test_redis_url_construction(self):
        """Test Redis URL is constructed correctly."""
        settings = get_settings()
        assert "redis://" in settings.redis_url


class TestCeleryApp:
    """Test Celery app creation."""

    def test_create_celery_app(self):
        """Test Celery app is created successfully."""
        app = create_celery_app("test-service")
        assert app is not None
        assert app.main == "test-service"

    def test_celery_config(self):
        """Test Celery configuration."""
        app = create_celery_app("test-service")
        assert app.conf.task_serializer == "json"
        assert app.conf.accept_content == ["json"]
        assert app.conf.result_serializer == "json"
        assert app.conf.timezone == "UTC"
        assert app.conf.enable_utc is True


class TestRedisConnection:
    """Test Redis connectivity."""

    @pytest.mark.skipif(
        not test_redis_connection(),
        reason="Redis not available",
    )
    def test_redis_connection(self):
        """Test Redis is reachable."""
        assert test_redis_connection() is True

    @pytest.mark.skipif(
        not test_redis_connection(),
        reason="Redis not available",
    )
    def test_redis_client_ping(self):
        """Test Redis client can ping."""
        client = get_redis_client()
        assert client.ping() is True
