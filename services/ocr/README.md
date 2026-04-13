# OCR

OCR service for Local Media Processor.

## Installation

```bash
pip install -e ".[dev]"
```

## Run

```bash
uvicorn services.ocr.app:app --host 0.0.0.0 --port 8000
celery -A services.ocr.worker worker --loglevel=info
```

