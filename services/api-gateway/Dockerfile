FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy shared package first (for dependency installation)
COPY services/shared /app/services/shared

# Install shared package
RUN pip install --no-cache-dir /app/services/shared

# Copy API Gateway service
COPY services/api-gateway /app/services/api-gateway

# Install API Gateway service
RUN pip install --no-cache-dir /app/services/api-gateway

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "services.api-gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
