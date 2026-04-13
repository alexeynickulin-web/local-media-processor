# Services

This directory contains all microservices for the Local Media Processor.

## Architecture

Each service is independent with:
- Its own FastAPI application
- Celery worker for async tasks
- Separate Docker image
- Individual dependencies

## Services

### Core Infrastructure
- **shared/** - Common components (config, models, utilities)
- **api-gateway/** - API Gateway (routes requests to services)
- **gradio-ui/** - Web interface

### Processing Services
- **transcription/** - Audio/video transcription (CUDA required)
- **translation/** - Text translation (CUDA required)
- **ocr/** - Image OCR (optional CUDA)
- **tts/** - Text-to-speech (CPU only)
- **subtitle/** - Subtitle generation (CPU only)

## Service Communication

```
Client → API Gateway → Redis Queue → Celery Worker → Service
                ↓
         Return task_id
                ↓
Client polls GET /api/status/{task_id}
```

## Development

### Install shared package locally:
```bash
cd services/shared
pip install -e ".[dev]"
```

### Run tests:
```bash
make test
# or
pytest services/
```

### Start development environment:
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### View logs:
```bash
make logs
make logs-transcription
make logs-translation
```

## Adding a New Service

1. Copy an existing service as template:
```bash
cp -r services/tts services/new-service
```

2. Update:
   - `pyproject.toml` (name, dependencies)
   - `app.py` (FastAPI routes)
   - `worker.py` (Celery tasks)
   - `Dockerfile` (base image)

3. Add to `docker-compose.yml`

4. Add route in `api-gateway/routes.py`

## Dependencies

Each service manages its own dependencies via `pyproject.toml`.
The `shared` package is installed in each service image.

## Testing

Tests are in each service's `tests/` directory:
```bash
pytest services/transcription/tests/
pytest services/translation/tests/
```

## Docker Images

- **CPU services**: `python:3.11-slim`
- **GPU services**: `nvidia/cuda:12.1.1-runtime-ubuntu22.04`

All containers run as non-root user `appuser`.
