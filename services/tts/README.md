# TTS

TTS service for Local Media Processor.

## Installation

```bash
pip install -e ".[dev]"
```

## Run

```bash
uvicorn services.tts.app:app --host 0.0.0.0 --port 8000
celery -A services.tts.worker worker --loglevel=info
```

