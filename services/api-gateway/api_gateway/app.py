"""API Gateway - FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .routes import router
from .rate_limiter import rate_limit_middleware
from .service_client import router as service_router
from shared.config import get_settings
from shared.logging_config import setup_logging
from shared.models import HealthResponse
from datetime import datetime

logger = setup_logging("api-gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("API Gateway starting up...")
    yield
    # Shutdown
    logger.info("API Gateway shutting down...")
    await service_router.close()


app = FastAPI(
    title="Local Media Processor - API Gateway",
    description="API Gateway for media processing services",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# Include routes
app.include_router(router, prefix="/api")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        service="api-gateway",
        status="healthy",
        timestamp=datetime.utcnow(),
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Local Media Processor - API Gateway",
        "version": "0.1.0",
        "docs": "/docs",
        "api_routes": "/api",
    }
