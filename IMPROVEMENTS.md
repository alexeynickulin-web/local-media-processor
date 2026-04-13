# Architecture Improvement Plan: Local Media Processor

## Project Overview

Local Media Processor is an offline, local-only web application for processing media files with:
- Audio/Video transcription (faster-whisper)
- OCR (EasyOCR)
- Text translation (HuggingFace NLLB)
- Text-to-Speech (edge_tts)
- Subtitle generation (pysrt)
- Batch processing with GPU support

## Current Architecture Analysis

### Strengths
- **Clear product concept**: Fully offline media processing pipeline is valuable
- **Good feature matrix**: Transcription → Translation → SRT → TTS workflow
- **Docker deployment**: GPU support with docker-compose
- **Comprehensive README**: Russian documentation with feature descriptions

### Critical Issues

#### 1. Single-File Monolith (app.py)
All code (400+ lines) lives in one file mixing:
- Infrastructure (model management, file I/O)
- Business logic (transcription, translation, OCR, TTS)
- Presentation (Gradio UI)

**Impact**: Untestable, unmaintainable, violates separation of concerns

#### 2. Silent Error Swallowing
Bare `except:` clauses in 6+ locations silently swallow all exceptions including `KeyboardInterrupt` and `SystemExit`

**Impact**: Impossible to debug, errors hidden from users and logs

#### 3. Conflicting Dependencies
Three different dependency specifications with conflicts:
- `requirements.txt` vs `pyproject.toml` have different packages
- `.python-version` says 3.11, Dockerfile uses 3.10-slim
- Unused packages listed (TTS, moviepy, yt_dlp)
- Missing packages in pyproject.toml (whisperx, edge_tts)

**Impact**: Build failures, runtime import errors

#### 4. Security Vulnerabilities
- **No authentication**: Gradio server on 0.0.0.0:7860 with no auth
- **Path traversal**: User can specify any filesystem path
- **Root container**: Docker runs as root user
- **No integrity checks**: Model downloads without checksum verification
- **TOCTOU race**: Temp file creation vulnerable to race conditions

**Impact**: Unauthorized access, arbitrary file access, container escape

#### 5. Performance Anti-Patterns
- **Model unload per file**: `mm.unload_all()` called after EVERY file in batch
- **Synchronous only**: No parallel processing of independent files
- **Text truncation**: `full_text[:3000]` silently discards data
- **No result caching**: Same file processed twice runs full pipeline
- **Heavy imports at startup**: All 14+ libraries loaded at module load

**Impact**: 10x slower batch processing, silent data loss, slow startup

#### 6. Missing Engineering Practices
- Zero tests
- Zero CI/CD
- Zero type hints
- Zero docstrings
- No code quality tooling configuration
- No configuration management (all hardcoded)

## Target Architecture: Microservices

### Why Microservices?
- **Independent scaling**: Transcription needs more GPU than TTS
- **Fault isolation**: OCR crash doesn't kill transcription
- **Different resource needs**: Some services need CUDA, others CPU-only
- **Long-running processes**: Transcription/translation benefit from async queues
- **Team scalability**: Different developers can work on services independently

### Proposed Microservices Architecture

```
                    ┌─────────────┐
                    │   Client    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ API Gateway │  (FastAPI, auth, rate limiting)
                    │  port 8000  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────────────┐
              │            │                    │
     ┌────────▼───┐  ┌────▼────┐          ┌────▼────┐
     │  Gradio UI │  │  Redis  │◄─────────┤ Celery  │
     │  port 7860 │  │ Queue   │─────────►│ Workers │
     └────────────┘  └─────────┘          └────┬────┘
                                               │
                    ┌──────────────────────────┼──────────────┐
                    │                          │              │
           ┌────────▼────────┐    ┌───────────▼──┐  ┌───────▼──────┐
           │ Transcription   │    │ Translation  │  │     OCR      │
           │ (CUDA, GPU)     │    │ (CUDA, GPU)  │  │ (opt. CUDA)  │
           │ port 8001       │    │ port 8002    │  │ port 8003    │
           └─────────────────┘    └──────────────┘  └──────────────┘
                                                     
           ┌─────────────────┐    ┌──────────────┐
           │      TTS        │    │   Subtitle   │
           │ (CPU only)      │    │ (CPU only)   │
           │ port 8004       │    │ port 8005    │
           └─────────────────┘    └──────────────┘
```

### Directory Structure

```
local-media-processor/
├── services/
│   ├── shared/                    # Shared package
│   │   ├── __init__.py
│   │   ├── config.py              # Shared configuration
│   │   ├── celery_app.py          # Celery app factory
│   │   ├── models.py              # Pydantic request/response models
│   │   ├── exceptions.py          # Shared error types
│   │   ├── file_io.py             # File utilities
│   │   ├── language.py            # Language codes/validation
│   │   └── logging_config.py      # Logging setup
│   │
│   ├── api-gateway/               # API Gateway service
│   │   ├── app.py                 # FastAPI application
│   │   ├── routes.py              # Route definitions
│   │   ├── middleware.py          # Auth, rate limiting
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── tests/
│   │
│   ├── gradio-ui/                 # Gradio UI service
│   │   ├── app.py                 # Gradio interface
│   │   ├── client.py              # API gateway client
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── tests/
│   │
│   ├── transcription/             # Transcription service
│   │   ├── app.py                 # FastAPI service
│   │   ├── worker.py              # Celery tasks
│   │   ├── model_manager.py       # Whisper model management
│   │   ├── Dockerfile             # CUDA base image
│   │   ├── pyproject.toml
│   │   └── tests/
│   │
│   ├── translation/               # Translation service
│   │   ├── app.py                 # FastAPI service
│   │   ├── worker.py              # Celery tasks
│   │   ├── model_manager.py       # NLLB model management
│   │   ├── chunker.py             # Text chunking for long texts
│   │   ├── Dockerfile             # CUDA base image
│   │   ├── pyproject.toml
│   │   └── tests/
│   │
│   ├── ocr/                       # OCR service
│   │   ├── app.py                 # FastAPI service
│   │   ├── worker.py              # Celery tasks
│   │   ├── model_manager.py       # EasyOCR model management
│   │   ├── Dockerfile             # Optional CUDA image
│   │   ├── pyproject.toml
│   │   └── tests/
│   │
│   ├── tts/                       # TTS service
│   │   ├── app.py                 # FastAPI service
│   │   ├── worker.py              # Celery tasks
│   │   ├── voice_map.py           # Language-to-voice mapping
│   │   ├── Dockerfile             # CPU-only image
│   │   ├── pyproject.toml
│   │   └── tests/
│   │
│   └── subtitle/                  # Subtitle service
│       ├── app.py                 # FastAPI service
│       ├── worker.py              # Celery tasks
│       ├── srt_generator.py       # pysrt logic
│       ├── Dockerfile             # Lightweight CPU image
│       ├── pyproject.toml
│       └── tests/
│
├── docker-compose.yml             # Multi-service orchestration
├── docker-compose.dev.yml         # Development overrides
├── .dockerignore
├── .env.example
├── .pre-commit-config.yaml
├── ruff.toml
├── Makefile
├── README.md
└── IMPROVEMENTS.md
```

### Service Communication Flow

```
User uploads file via Gradio UI
  ↓
Gradio UI → POST /api/transcribe → API Gateway
  ↓
API Gateway → Redis Queue → Celery Worker
  ↓
Transcription Service processes file (async)
  ↓
Gradio UI polls GET /api/status/{task_id}
  ↓
Returns result when complete
```

### Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| API Framework | FastAPI | Async support, auto OpenAPI docs, performance |
| UI | Gradio | Keep existing UI, minimal migration effort |
| Task Queue | Celery + Redis | Mature, retries, monitoring, Python-native |
| Message Broker | Redis | Fast, persistent, Celery integration |
| Result Backend | Redis | Task results, caching |
| Container Orchestration | Docker Compose | Simple local dev, production-ready |
| Model Storage | Shared volume | All services access `./models` |

### Key Architectural Decisions

#### 1. API Gateway Pattern
- Single entry point for all requests
- Centralized auth, rate limiting, validation
- Routes to appropriate backend services
- Task status aggregation

#### 2. Async Task Processing
- Long operations (transcription, translation) use Celery
- Submit task → get task_id → poll for results
- Prevents HTTP timeout on long operations
- Built-in retry logic for failures

#### 3. Service Independence
- Each service has its own model manager
- Services can be scaled independently
- Different Docker images (CUDA vs CPU)
- Failure in one doesn't affect others

#### 4. Shared Components
- `services/shared/` package for common code
- Pydantic models for consistent APIs
- Shared logging with request ID tracing
- Common configuration management

#### 5. Configuration Management
- Environment variables per service
- Shared config in `services/shared/config.py`
- `.env.example` with all defaults
- Pydantic Settings for type-safe config

## Migration Strategy

### Strangler Fig Pattern
Instead of rewriting everything at once:

1. **Phase 0**: Set up infrastructure (Redis, Celery, API gateway)
2. **Phase 1**: Extract services one by one, test independently
3. **Phase 2**: Migrate UI to use API gateway, deprecate direct calls
4. **Phase 3**: Add production readiness (auth, monitoring, tests)
5. **Phase 4**: Optimize and enhance

**During migration:**
- Original `app.py` remains functional as fallback
- New services tested independently first
- UI switches to API gateway incrementally
- Old code removed only after new services verified

## Implementation Phases

### Phase 0: Foundation (Week 1-2)
- Create project structure
- Set up Redis + Celery infrastructure
- Create API Gateway service

### Phase 1: Extract Core Services (Week 3-5)
- Extract Transcription service
- Extract Translation service
- Extract OCR service
- Extract TTS service
- Extract Subtitle service

### Phase 2: UI and Integration (Week 6-7)
- Refactor Gradio UI to use API gateway
- Implement shared components
- Create Docker Compose setup

### Phase 3: Quality and Production (Week 8-9)
- Add authentication & security
- Add monitoring & observability
- Add testing infrastructure
- Add CI/CD pipeline
- Add code quality tooling

### Phase 4: Optimization (Week 10+)
- Implement caching layer
- Add advanced queue features
- Performance optimization
- Documentation & developer experience

## Success Metrics

- **Service independence**: Each service can run/deploy independently
- **Task queue**: Long operations handled asynchronously
- **Test coverage**: >80% per service
- **Scalability**: Services can be scaled independently
- **Security**: Zero critical vulnerabilities
- **Code quality**: Clean ruff/mypy checks
- **Developer experience**: New service scaffold in <5 min
- **Performance**: 5x faster batch processing (no model reload)

## Benefits of Microservices Approach

| Aspect | Monolith | Microservices |
|--------|----------|---------------|
| Scaling | Scale entire app | Scale only needed services |
| GPU Usage | All services get GPU | Only CUDA services get GPU |
| Fault Isolation | One crash kills all | Services fail independently |
| Development | Single team bottleneck | Parallel development |
| Testing | Hard to isolate | Easy to test per service |
| Deployment | All-or-nothing | Independent deployments |
| Resources | Wasteful | Right-sized per service |

