"""Service client for communication with backend services."""

import httpx
from typing import Any
from shared.config import get_settings
from shared.logging_config import setup_logging
from shared.exceptions import ProcessingError, TimeoutError

logger = setup_logging("service-client")


class ServiceClient:
    """HTTP client for communicating with backend services."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        retries: int = 3,
    ):
        """Initialize service client.

        Args:
            base_url: Base URL of the service
            timeout: Request timeout in seconds
            retries: Number of retries on failure
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with retries.

        Args:
            method: HTTP method
            path: Request path
            json: JSON body
            params: Query parameters

        Returns:
            Response JSON

        Raises:
            ProcessingError: If request fails after retries
            TimeoutError: If request times out
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error = None

        for attempt in range(self.retries):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    json=json,
                    params=params,
                )
                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Request to {url} timed out (attempt {attempt + 1}/{self.retries})")
                if attempt == self.retries - 1:
                    raise TimeoutError(f"Request to {url} timed out after {self.retries} attempts") from e

            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx)
                if e.response.status_code < 500:
                    raise ProcessingError(f"Client error: {e.response.status_code} - {e.response.text}") from e
                last_error = e
                logger.warning(f"Server error {e.response.status_code} (attempt {attempt + 1}/{self.retries})")

            except Exception as e:
                last_error = e
                logger.warning(f"Request to {url} failed (attempt {attempt + 1}/{self.retries}): {e}")

        raise ProcessingError(f"Request to {url} failed after {self.retries} attempts: {last_error}")

    async def post(self, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a POST request."""
        return await self._request("POST", path, json=json)

    async def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a GET request."""
        return await self._request("GET", path, params=params)

    async def health_check(self) -> bool:
        """Check if the service is healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed for {self.base_url}: {e}")
            return False


class ServiceRouter:
    """Routes requests to appropriate backend services."""

    def __init__(self):
        """Initialize service router."""
        settings = get_settings()
        self.services = {
            "transcription": ServiceClient(settings.transcription_service_url),
            "translation": ServiceClient(settings.translation_service_url),
            "ocr": ServiceClient(settings.ocr_service_url),
            "tts": ServiceClient(settings.tts_service_url),
            "subtitle": ServiceClient(settings.subtitle_service_url),
        }

    async def close(self):
        """Close all service clients."""
        for client in self.services.values():
            await client.close()

    def get_client(self, service_name: str) -> ServiceClient:
        """Get client for a specific service.

        Args:
            service_name: Name of the service

        Returns:
            ServiceClient instance

        Raises:
            ProcessingError: If service not found
        """
        client = self.services.get(service_name)
        if not client:
            raise ProcessingError(f"Unknown service: {service_name}")
        return client

    async def check_all_services(self) -> dict[str, bool]:
        """Check health of all services.

        Returns:
            Dict mapping service names to health status
        """
        import asyncio
        async def check_service(name: str, client: ServiceClient) -> tuple[str, bool]:
            return name, await client.health_check()

        tasks = [check_service(name, client) for name, client in self.services.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        health_status = {}
        for result in results:
            if isinstance(result, Exception):
                health_status["unknown"] = False
            else:
                name, healthy = result
                health_status[name] = healthy

        return health_status


# Global service router instance
router = ServiceRouter()
