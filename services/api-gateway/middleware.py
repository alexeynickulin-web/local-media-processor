"""API Gateway middleware."""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import time
import uuid
from shared.logging_config import setup_logging

logger = setup_logging("api-gateway-middleware")


class RequestLoggingMiddleware:
    """Middleware for request logging."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        scope["state"] = scope.get("state", {})
        scope["state"]["request_id"] = request_id

        start_time = time.time()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                process_time = time.time() - start_time
                message.setdefault("headers", []).append(
                    (b"x-request-id", request_id.encode())
                )
                message.setdefault("headers", []).append(
                    (b"x-process-time", str(process_time).encode())
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)


class ErrorHandlerMiddleware:
    """Middleware for centralized error handling."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        try:
            await self.app(scope, receive, send)
        except MediaProcessorError as e:
            response = JSONResponse(
                status_code=400,
                content={"error": str(e), "type": type(e).__name__},
            )
            await response(scope, receive, send)
        except Exception as e:
            logger.error(f"Unhandled exception: {e}", exc_info=True)
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )
            await response(scope, receive, send)


def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware on FastAPI app."""
    app.middleware("http")(RequestLoggingMiddleware)
    # Add more middleware as needed
