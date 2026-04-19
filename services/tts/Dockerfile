FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy shared package
COPY services/shared /app/services/shared
RUN pip install --no-cache-dir -e /app/services/shared

# Copy TTS service
COPY services/tts /app/services/tts

# Install dependencies directly
RUN pip install --no-cache-dir \
    "fastapi>=0.109.0" \
    "uvicorn[standard]>=0.27.0" \
    "edge-tts>=6.1.0"

# Create output directory before switching user
RUN mkdir -p /app/output /app/media && chmod 777 /app/output /app/media

# Copy and set up start script before switching user
COPY services/tts/start.sh /start.sh
RUN chmod +x /start.sh

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/start.sh"]