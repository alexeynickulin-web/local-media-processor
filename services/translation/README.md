# Translation

Translation service for Local Media Processor.

## Installation

```bash
pip install -e ".[dev]"
```

## Run

```bash
uvicorn services.translation.app:app --host 0.0.0.0 --port 8000
celery -A services.translation.worker worker --loglevel=info
```

