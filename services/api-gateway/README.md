# API Gateway

API Gateway service for Local Media Processor. Routes requests to backend services.

## Installation

```bash
pip install -e ".[dev]"
```

## Run

```bash
uvicorn services.api_gateway.app:app --host 0.0.0.0 --port 8000
```
