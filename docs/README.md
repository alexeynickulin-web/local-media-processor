# Local Media Processor

Microservices-based application for processing media files locally: audio/video transcription, OCR, translation, TTS — all on your machine, no cloud APIs.

Features a **Gradio** web interface, runs in **Docker** with support for GPU and CPU.

[https://github.com/alexeynickulin-web/local-media-processor](https://github.com/alexeynickulin-web/local-media-processor)

## Features

- Audio/Video transcription (faster-whisper + whisperx)
- Image OCR (EasyOCR)
- Text translation (NLLB - facebook/nllb-200-distilled-600M)
- Text-to-Speech (edge-tts)
- Subtitle generation (SRT/VTT)
- Flexible: run single step or full pipeline
- Async task processing via Celery + Redis
- GPU (CUDA) and CPU support
- All models stored locally in `./models`

## Architecture

```
gradio-ui (7860) → api-gateway (8000) → redis (6379)
                                              ↓
                        ┌──────────────────────┼──────────────────────┐
                        ↓                      ↓                      ↓
              transcription (GPU)      translation (GPU)      ocr (CPU)
                        ↓                      ↓                      ↓
                        tts (CPU)              subtitle (CPU)
```

## Requirements

- Docker + Docker Compose
- NVIDIA GPU + drivers + CUDA toolkit (optional, for acceleration)
- ≥ 8 GB RAM (recommended 16+ GB)
- Free disk space: ~15–25 GB for models (first run)

## Quick Start

1. Clone repository

```bash
git clone https://github.com/your-username/local-media-processor.git
cd local-media-processor
```

2. (Optional) Create `.env` for configuration

```bash
# .env
REDIS_URL=redis://redis:6379/0
API_GATEWAY_URL=http://api-gateway:8000
```

3. Build and run

```bash
docker-compose up -d
```

4. Open in browser:
- Gradio UI: http://localhost:7860
- API Gateway: http://localhost:8000

5. Upload files and process!

## Services

| Service       | Port | Description                      |
|--------------|-----|----------------------------------|
| Gradio UI    | 7860 | Web interface                   |
| API Gateway  | 8000 | Routes requests to services     |
| Redis       | 6379 | Message broker + result backend|
| Transcription | 8001 | Whisper + WhisperX          |
| Translation | 8002 | NLLB translation            |
| OCR        | 8003 | EasyOCR                     |
| TTS       | 8004 | edge-tts                    |
| Subtitle  | 8005 | SRT/VTT generation          |

## Useful Commands

```bash
# Start all services
docker-compose up -d

# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api-gateway
docker-compose logs -f transcription

# Check health of all services
make health

# Stop all services
docker-compose down

# Clean up (including volumes)
docker-compose down -v
```

## Project Structure

```text
local-media-processor/
├── docker-compose.yml      # Multi-service setup
├── Makefile                 # Development commands
├── docs/                   # Documentation
│   ├── README.md
│   └── TODO.md
├── services/
│   ├── shared/            # Shared package (Celery, models, config)
│   ├── api-gateway/       # API Gateway (routes, auth, rate limiting)
│   ├── gradio-ui/         # Gradio Web UI
│   ├── transcription/     # Transcription service (Whisper)
│   ├── translation/       # Translation service (NLLB)
│   ├── ocr/             # OCR service (EasyOCR)
│   ├── tts/             # TTS service (edge-tts)
│   └── subtitle/        # Subtitle service (pysrt)
└── models/               # All downloaded models
```

## Development

```bash
# Run linting
make lint

# Fix linting issues
make lint-fix

# Run tests
make test

# Build all images
make build

# Open shell in service container
make shell-transcription
make shell-gradio
```

## API Usage

Submit task:

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/file.mp3", "model_name": "medium"}'
```

Get task status:

```bash
curl http://localhost:8000/status/{task_id}
```

---

*Last updated: 2026-04-19*