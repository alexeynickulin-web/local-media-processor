# Shared Components

Common components for all Local Media Processor microservices.

## Components

- **config.py** - Shared settings and configuration
- **celery_app.py** - Celery application factory
- **models.py** - Pydantic request/response models
- **exceptions.py** - Shared error types
- **file_io.py** - File utilities
- **language.py** - Language codes and validation
- **logging_config.py** - Logging configuration
- **redis_client.py** - Redis client and utilities

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from shared import get_settings, create_celery_app, setup_logging

settings = get_settings()
logger = setup_logging("my-service")
celery_app = create_celery_app("my-service")
```
