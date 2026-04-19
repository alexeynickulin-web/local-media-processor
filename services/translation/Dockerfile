FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install Python and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy shared package
COPY services/shared /app/services/shared
RUN pip3 install --no-cache-dir /app/services/shared

# Copy Translation service
COPY services/translation /app/services/translation
RUN pip3 install --no-cache-dir /app/services/translation

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn services.translation.app:app --host 0.0.0.0 --port 8000 & celery -A services.translation.worker worker --loglevel=info -Q translation"]
