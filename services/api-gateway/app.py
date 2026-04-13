"""API Gateway - FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from .middleware import setup_middleware
from shared.config import get_settings
from shared.logging_config import setup_logging
from shared.models import HealthResponse
from datetime import datetime

logger = setup_logging("api-gateway")

app = FastAPI(
    title="Local Media Processor - API Gateway",
    description="API Gateway for media processing services",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup additional middleware
setup_middleware(app)

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
    }
