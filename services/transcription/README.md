# Transcription

Transcription service for Local Media Processor.

## Installation

```bash
pip install -e ".[dev]"
```

## Run

```bash
uvicorn services.transcription.app:app --host 0.0.0.0 --port 8000
celery -A services.transcription.worker worker --loglevel=info
```

