# Subtitle

Subtitle service for Local Media Processor.

## Installation

```bash
pip install -e ".[dev]"
```

## Run

```bash
uvicorn services.subtitle.app:app --host 0.0.0.0 --port 8000
celery -A services.subtitle.worker worker --loglevel=info
```

