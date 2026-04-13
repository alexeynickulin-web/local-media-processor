"""Rate limiting middleware for API Gateway."""

import time
from collections import defaultdict
from fastapi import Request, HTTPException, status
from shared.logging_config import setup_logging

logger = setup_logging("rate-limiter")


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests: dict[str, list[float]] = defaultdict(list)

    def _cleanup_old_requests(self, client_id: str, now: float):
        """Remove requests older than 1 hour.

        Args:
            client_id: Client identifier
            now: Current timestamp
        """
        cutoff = now - 3600  # 1 hour ago
        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > cutoff
        ]

    def is_allowed(self, client_id: str) -> bool:
        """Check if a request is allowed for the client.

        Args:
            client_id: Client identifier (e.g., IP address)

        Returns:
            True if request is allowed
        """
        now = time.time()
        self._cleanup_old_requests(client_id, now)

        # Check per-minute limit
        one_minute_ago = now - 60
        recent_requests = [t for t in self.requests[client_id] if t > one_minute_ago]
        if len(recent_requests) >= self.requests_per_minute:
            return False

        # Check per-hour limit
        if len(self.requests[client_id]) >= self.requests_per_hour:
            return False

        # Record this request
        self.requests[client_id].append(now)
        return True

    def get_retry_after(self, client_id: str) -> float:
        """Get seconds until next request is allowed.

        Args:
            client_id: Client identifier

        Returns:
            Seconds to wait
        """
        now = time.time()
        requests = self.requests.get(client_id, [])

        if not requests:
            return 0

        # Check if we hit per-minute limit
        one_minute_ago = now - 60
        recent = [t for t in requests if t > one_minute_ago]
        if len(recent) >= self.requests_per_minute:
            oldest = min(recent)
            return max(0, 60 - (now - oldest))

        # Check if we hit per-hour limit
        if len(requests) >= self.requests_per_hour:
            oldest = min(requests)
            return max(0, 3600 - (now - oldest))

        return 0


# Global rate limiter instance
rate_limiter = RateLimiter(
    requests_per_minute=60,
    requests_per_hour=1000,
)


async def rate_limit_middleware(request: Request, call_next):
    """FastAPI middleware for rate limiting.

    Args:
        request: FastAPI request
        call_next: Next middleware

    Returns:
        Response
    """
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"

    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)

    # Check rate limit
    if not rate_limiter.is_allowed(client_ip):
        retry_after = rate_limiter.get_retry_after(client_ip)
        logger.warning(f"Rate limit exceeded for {client_ip}")

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(retry_after))},
        )

    # Process request
    response = await call_next(request)

    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.requests_per_minute)
    response.headers["X-RateLimit-Remaining"] = str(
        max(0, rate_limiter.requests_per_minute - len(rate_limiter.requests.get(client_ip, [])))
    )

    return response
