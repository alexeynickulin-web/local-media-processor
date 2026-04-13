"""Authentication middleware for API Gateway."""

from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import time
import hashlib
import hmac
from shared.logging_config import setup_logging
from shared.exceptions import MediaProcessorError

logger = setup_logging("auth-middleware")

# Security scheme
security = HTTPBearer(auto_error=False)


class AuthenticationError(MediaProcessorError):
    """Raised when authentication fails."""

    pass


class APIKeyManager:
    """Manages API key validation."""

    def __init__(self):
        """Initialize API key manager."""
        self.api_keys = self._load_api_keys()

    def _load_api_keys(self) -> set[str]:
        """Load API keys from environment variable.

        Returns:
            Set of valid API keys
        """
        keys_env = os.getenv("API_KEYS", "")
        if not keys_env:
            return set()

        return {key.strip() for key in keys_env.split(",") if key.strip()}

    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key.

        Args:
            api_key: API key to validate

        Returns:
            True if valid
        """
        if not self.api_keys:
            # No keys configured, skip authentication
            return True
        return api_key in self.api_keys


class JWTManager:
    """Manages JWT token validation (simplified implementation)."""

    def __init__(self):
        """Initialize JWT manager."""
        self.secret_key = os.getenv("JWT_SECRET_KEY", "")
        self.enabled = bool(self.secret_key)

    def validate_token(self, token: str) -> dict | None:
        """Validate a JWT token.

        Args:
            token: JWT token

        Returns:
            Token payload if valid, None otherwise
        """
        if not self.enabled:
            return None

        # TODO: Implement full JWT validation
        # For now, just check if secret key is set
        try:
            import jwt
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except Exception as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None


# Global managers
api_key_manager = APIKeyManager()
jwt_manager = JWTManager()


async def verify_api_key(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> str | None:
    """Verify API key from request.

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        API key if valid, None if no authentication required

    Raises:
        HTTPException: If authentication fails
    """
    # Check if authentication is required
    api_keys = api_key_manager.api_keys
    if not api_keys and not jwt_manager.enabled:
        # No authentication configured
        return None

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Try API key validation first
    if api_key_manager.validate_api_key(credentials.credentials):
        return credentials.credentials

    # Try JWT validation
    if jwt_manager.enabled:
        payload = jwt_manager.validate_token(credentials.credentials)
        if payload:
            return payload.get("sub")

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid authentication credentials",
    )


async def verify_request_signature(
    request: Request,
    x_signature: str | None = None,
) -> bool:
    """Verify request body signature (for webhook validation).

    Args:
        request: FastAPI request
        x_signature: Signature from X-Signature header

    Returns:
        True if signature valid
    """
    if not x_signature:
        return True  # No signature required

    secret = os.getenv("WEBHOOK_SECRET", "")
    if not secret:
        return True

    body = await request.body()
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(x_signature, expected):
        logger.warning("Invalid request signature")
        return False

    return True


def require_auth(enabled: bool = True):
    """Dependency factory for requiring authentication.

    Args:
        enabled: Whether authentication is required

    Returns:
        Dependency function
    """
    if not enabled:
        return lambda: None

    return verify_api_key
