#!/bin/bash
set -e

echo "Starting TTS service..."

# Start uvicorn in background
echo "Starting uvicorn..."
uvicorn services.tts.app:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

sleep 2

# Start Celery worker
echo "Starting Celery worker..."
exec celery -A services.tts.worker worker --loglevel=info -Q tts