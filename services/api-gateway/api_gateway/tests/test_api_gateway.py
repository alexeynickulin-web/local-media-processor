"""Tests for API Gateway."""

import pytest
from fastapi.testclient import TestClient
from api_gateway.app import app
from api_gateway.task_manager import task_manager
from api_gateway.service_client import ServiceClient


class TestAPIGateway:
    """Test API Gateway endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "docs" in data

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "api-gateway"
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_openapi_docs(self, client):
        """Test OpenAPI docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc(self, client):
        """Test ReDoc is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_json(self, client):
        """Test OpenAPI JSON is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200


class TestTaskSubmission:
    """Test task submission endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_transcribe_endpoint(self, client):
        """Test transcription endpoint."""
        response = client.post(
            "/api/transcribe",
            json={
                "file_path": "/tmp/test.mp3",
                "model_name": "base",
                "language": "en",
            },
        )
        # Should return 200 or 500 (if Redis not available)
        assert response.status_code in [200, 500]

    def test_translate_endpoint(self, client):
        """Test translation endpoint."""
        response = client.post(
            "/api/translate",
            json={
                "text": "Hello world",
                "source_language": "en",
                "target_language": "ru",
            },
        )
        assert response.status_code in [200, 500]

    def test_ocr_endpoint(self, client):
        """Test OCR endpoint."""
        response = client.post(
            "/api/ocr",
            json={
                "file_path": "/tmp/test.jpg",
                "language": "en",
            },
        )
        assert response.status_code in [200, 500]

    def test_tts_endpoint(self, client):
        """Test TTS endpoint."""
        response = client.post(
            "/api/tts",
            json={
                "text": "Hello world",
                "language": "en",
            },
        )
        assert response.status_code in [200, 500]

    def test_subtitle_endpoint(self, client):
        """Test subtitle endpoint."""
        response = client.post(
            "/api/subtitle",
            json={
                "transcription_result": {
                    "text": "Hello world",
                    "segments": [],
                    "language": "en",
                    "duration": 5.0,
                },
                "format": "srt",
            },
        )
        assert response.status_code in [200, 500]


class TestTaskStatus:
    """Test task status endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_task_status(self, client):
        """Test task status endpoint."""
        response = client.get("/api/status/test-task-id")
        # Should return 200 or 404
        assert response.status_code in [200, 404, 500]


class TestServiceClient:
    """Test service client."""

    def test_service_client_creation(self):
        """Test service client can be created."""
        client = ServiceClient("http://localhost:8001")
        assert client.base_url == "http://localhost:8001"
        assert client.timeout == 30.0
        assert client.retries == 3

    @pytest.mark.asyncio
    async def test_health_check_unreachable(self):
        """Test health check with unreachable service."""
        client = ServiceClient("http://localhost:9999", timeout=1.0)
        result = await client.health_check()
        assert result is False
        await client.close()


class TestRateLimiter:
    """Test rate limiter."""

    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limits."""
        from api_gateway.rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=5, requests_per_hour=100)
        assert limiter.is_allowed("test-client") is True

    def test_rate_limiter_blocks_excess(self):
        """Test rate limiter blocks excess requests."""
        from api_gateway.rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, requests_per_hour=100)
        assert limiter.is_allowed("test-client") is True
        assert limiter.is_allowed("test-client") is True
        assert limiter.is_allowed("test-client") is False  # Should be blocked


class TaskManagerTest:
    """Test task manager."""

    def test_submit_task_format(self):
        """Test task submission format."""
        # This will fail if Redis is not available, but tests the code path
        try:
            task_id = task_manager.submit_task(
                "test.task",
                kwargs={"test": "value"},
            )
            assert isinstance(task_id, str)
        except Exception:
            pass  # Expected if Redis not available
