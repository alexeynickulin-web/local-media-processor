FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy shared package
COPY services/shared /app/services/shared
RUN pip install --no-cache-dir /app/services/shared

# Copy TTS service
COPY services/tts /app/services/tts
RUN pip install --no-cache-dir /app/services/tts

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn services.tts.app:app --host 0.0.0.0 --port 8000 & celery -A services.tts.worker worker --loglevel=info"]
