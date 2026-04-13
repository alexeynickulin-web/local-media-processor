# TODO: Local Media Processor - Microservices Migration

This file tracks the migration from monolithic app.py to microservices architecture.
Tasks are organized in phases for sequential implementation.

---

## Phase 0: Foundation (Week 1-2)

### 1. Create Project Structure
**Problem**: Single-file monolith needs to become multi-service architecture
**Steps**:
- [ ] Create root workspace structure:
  ```
  local-media-processor/
  ├── services/
  │   ├── api-gateway/
  │   ├── gradio-ui/
  │   ├── transcription/
  │   ├── translation/
  │   ├── ocr/
  │   ├── tts/
  │   └── shared/
  ├── docker-compose.yml
  └── README.md
  ```
- [ ] Create `services/shared/` package with common types, utils, config
- [ ] Set up individual service directories with standard FastAPI structure
- [ ] Create base Dockerfile template for services
- [ ] Update root docker-compose.yml for multi-service setup

### 2. Set Up Redis + Celery Infrastructure
**Files**: `docker-compose.yml`, shared Celery config
**Steps**:
- [ ] Add Redis service to docker-compose.yml
- [ ] Add Redis configuration with persistence
- [ ] Create shared Celery app configuration in `services/shared/celery_app.py`
- [ ] Create base Celery worker template
- [ ] Configure Redis as message broker and result backend
- [ ] Test Redis connectivity from services
- [ ] Add Redis monitoring/health check

### 3. Create API Gateway Service
**Directory**: `services/api-gateway/`
**Framework**: FastAPI
**Steps**:
- [ ] Create FastAPI application with basic routing
- [ ] Implement authentication middleware (JWT or API key)
- [ ] Add rate limiting middleware
- [ ] Create route table:
  - `/api/transcribe` → transcription service
  - `/api/translate` → translation service
  - `/api/ocr` → OCR service
  - `/api/tts` → TTS service
  - `/api/batch` → batch processing
  - `/api/status/{task_id}` → task status
- [ ] Add request validation and sanitization
- [ ] Implement centralized error handling
- [ ] Add OpenAPI/Swagger docs
- [ ] Add health check endpoint `/health`
- [ ] Create Dockerfile for API gateway
- [ ] Test gateway routes

---

## Phase 1: Extract Core Services (Week 3-5)

### 4. Extract Transcription Service
**Directory**: `services/transcription/`
**Current**: `app.py` transcription logic
**Steps**:
- [ ] Create FastAPI service with `/transcribe` endpoint
- [ ] Move faster-whisper logic from app.py
- [ ] Move ModelManager (whisper part) to service
- [ ] Create Celery task for long transcription jobs
- [ ] Implement async task submission (return task_id)
- [ ] Add task status endpoint
- [ ] Add file upload handling (multipart or path)
- [ ] Add GPU configuration
- [ ] Create service-specific config
- [ ] Remove bare except clauses
- [ ] Add proper error handling and logging
- [ ] Create Dockerfile (CUDA base image)
- [ ] Add service tests
- [ ] Test with API gateway integration

### 5. Extract Translation Service
**Directory**: `services/translation/`
**Current**: `app.py` translation logic
**Steps**:
- [ ] Create FastAPI service with `/translate` endpoint
- [ ] Move NLLB translation logic from app.py
- [ ] Move ModelManager (NLLB part) to service
- [ ] Create Celery task for translation jobs
- [ ] Implement chunked translation (fix [:3000] truncation)
- [ ] Add language code validation
- [ ] Add GPU/CPU configuration
- [ ] Create service-specific config
- [ ] Remove bare except clauses
- [ ] Add proper error handling
- [ ] Create Dockerfile (CUDA base image)
- [ ] Add service tests
- [ ] Test with API gateway integration

### 6. Extract OCR Service
**Directory**: `services/ocr/`
**Current**: `app.py` OCR logic
**Steps**:
- [ ] Create FastAPI service with `/ocr` endpoint
- [ ] Move EasyOCR logic from app.py
- [ ] Move ModelManager (OCR part) to service
- [ ] Create Celery task for OCR jobs
- [ ] Add image format validation
- [ ] Add GPU/CPU configuration
- [ ] Create service-specific config
- [ ] Remove bare except clauses
- [ ] Add proper error handling
- [ ] Create Dockerfile (CUDA or CPU image)
- [ ] Add service tests
- [ ] Test with API gateway integration

### 7. Extract TTS Service
**Directory**: `services/tts/`
**Current**: `app.py` TTS logic
**Steps**:
- [ ] Create FastAPI service with `/tts` endpoint
- [ ] Move edge_tts logic from app.py
- [ ] Create Celery task for TTS jobs
- [ ] Add voice selection validation
- [ ] Add language-to-voice mapping
- [ ] Create service-specific config
- [ ] Remove bare except clauses
- [ ] Add proper error handling
- [ ] Create Dockerfile (CPU-only, no CUDA needed)
- [ ] Add service tests
- [ ] Test with API gateway integration

### 8. Extract Subtitle Service
**Directory**: `services/subtitle/`
**Current**: `app.py` SRT generation logic
**Steps**:
- [ ] Create FastAPI service with `/subtitle` endpoint
- [ ] Move pysrt logic from app.py
- [ ] Create Celery task for subtitle generation
- [ ] Add SRT format validation
- [ ] Create service-specific config
- [ ] Add proper error handling
- [ ] Create Dockerfile (lightweight, no GPU)
- [ ] Add service tests
- [ ] Test with API gateway integration

---

## Phase 2: UI and Integration (Week 6-7)

### 9. Refactor Gradio UI Service
**Directory**: `services/gradio-ui/`
**Current**: `app.py` UI logic
**Steps**:
- [ ] Create separate FastAPI + Gradio service
- [ ] Move Gradio interface from app.py
- [ ] Update UI to call API gateway instead of direct processing
- [ ] Implement async task polling (show progress)
- [ ] Add task cancellation support in UI
- [ ] Add batch upload with progress tracking
- [ ] Add auth configuration (env var GRADIO_AUTH)
- [ ] Update file input to work with API gateway
- [ ] Remove all processing logic from UI service
- [ ] Create Dockerfile
- [ ] Test complete UI flow with gateway
- [ ] Test batch processing through new architecture

### 10. Implement Shared Components
**Directory**: `services/shared/`
**Steps**:
- [ ] Create shared configuration module
- [ ] Create shared Celery app factory
- [ ] Create shared logging configuration
- [ ] Create shared file I/O utilities
- [ ] Create shared language codes/validation
- [ ] Create shared Pydantic models for requests/responses
- [ ] Create shared error types
- [ ] Add shared test utilities
- [ ] Package as installable Python package
- [ ] Add shared package tests

### 11. Create Docker Compose Setup
**File**: `docker-compose.yml`
**Steps**:
- [ ] Define all services:
  - api-gateway
  - gradio-ui
  - transcription (with GPU support)
  - translation (with GPU support)
  - ocr (optional GPU)
  - tts (CPU only)
  - subtitle (CPU only)
  - redis
  - redis-commander (optional, for debugging)
- [ ] Configure shared networks
- [ ] Configure volume mounts for:
  - Models directory
  - Input/output files
  - Redis data persistence
- [ ] Add service dependencies (depends_on)
- [ ] Add health checks for all services
- [ ] Add resource limits (memory, GPU)
- [ ] Create development and production compose files
- [ ] Test full stack startup
- [ ] Document docker-compose usage in README

---

## Phase 3: Quality and Production (Week 8-9)

### 12. Add Authentication & Security
**Steps**:
- [ ] Implement JWT or API key auth in API gateway
- [ ] Add auth middleware to protect endpoints
- [ ] Configure Gradio auth for UI access
- [ ] Add path validation and sanitization in all services
- [ ] Run all Docker containers as non-root users
- [ ] Add checksum verification for model downloads
- [ ] Add rate limiting in API gateway
- [ ] Add request size limits
- [ ] Test auth flows end-to-end

### 13. Add Monitoring & Observability
**Steps**:
- [ ] Add structured logging (JSON) to all services
- [ ] Add request ID tracing across services
- [ ] Add health check endpoints to all services
- [ ] Add metrics collection (Prometheus or similar)
- [ ] Add task monitoring (Celery flower or custom dashboard)
- [ ] Add error tracking (Sentry or similar)
- [ ] Create monitoring dashboard
- [ ] Add log aggregation
- [ ] Document monitoring setup

### 14. Add Testing Infrastructure
**Directory**: `tests/` in each service
**Steps**:
- [ ] Set up pytest in each service
- [ ] Add unit tests for each service
- [ ] Add integration tests for API gateway routing
- [ ] Add end-to-end tests for full workflows
- [ ] Add Celery task testing
- [ ] Add Docker Compose integration tests
- [ ] Aim for >80% coverage per service
- [ ] Add test reporting

### 15. Add CI/CD Pipeline
**File**: `.github/workflows/ci.yml`
**Steps**:
- [ ] Create GitHub Actions workflow
- [ ] Add Python linting (ruff) for each service
- [ ] Add type checking (mypy) for each service
- [ ] Add pytest runs for each service
- [ ] Add Docker image builds
- [ ] Add Docker image pushes to registry
- [ ] Add integration test runs
- [ ] Add status badge to README

### 16. Add Code Quality Tooling
**Files**: `.pre-commit-config.yaml`, `ruff.toml`
**Steps**:
- [ ] Add pre-commit hooks (ruff, mypy)
- [ ] Add ruff configuration
- [ ] Add mypy configuration
- [ ] Add trailing whitespace fixes
- [ ] Test pre-commit on all services
- [ ] Document code quality standards

---

## Phase 4: Optimization (Week 10+)

### 17. Implement Caching Layer
**Steps**:
- [ ] Use Redis for result caching
- [ ] Design cache key strategy (file hash + params)
- [ ] Add cache lookup before processing
- [ ] Add cache invalidation logic
- [ ] Make cache TTL configurable
- [ ] Add cache size limits
- [ ] Document caching behavior
- [ ] Benchmark cache hit performance

### 18. Add Advanced Queue Features
**Steps**:
- [ ] Implement task priority queues
- [ ] Add task cancellation support
- [ ] Add task retry logic for failures
- [ ] Add batch processing orchestration
- [ ] Add progress persistence
- [ ] Add queue monitoring dashboard
- [ ] Add dead letter queue for failed tasks
- [ ] Document queue behavior

### 19. Performance Optimization
**Steps**:
- [ ] Benchmark each service individually
- [ ] Optimize model loading/caching
- [ ] Implement GPU memory management
- [ ] Add request batching where possible
- [ ] Add connection pooling (Redis, DB)
- [ ] Optimize Docker image sizes
- [ ] Add CDN for static assets (if any)
- [ ] Load test full stack

### 20. Documentation & Developer Experience
**Steps**:
- [ ] Add English README translation
- [ ] Create architecture diagrams
- [ ] Document each service API
- [ ] Create getting started guide
- [ ] Add troubleshooting guide
- [ ] Create Makefile with common commands
- [ ] Add .env.example files
- [ ] Document deployment process

---

## Migration Strategy

### Approach: Strangler Fig Pattern
Instead of rewriting everything at once, we'll gradually extract services while keeping the existing app.py functional during transition:

1. **Week 1-2**: Set up infrastructure (Redis, Celery, API gateway)
2. **Week 3-5**: Extract services one by one, test each independently
3. **Week 6-7**: Migrate UI to use new services, deprecate direct calls
4. **Week 8-9**: Add production readiness (auth, monitoring, tests)
5. **Week 10+**: Optimize and enhance

During migration:
- Original `app.py` remains functional as fallback
- New services are tested independently first
- UI switches to API gateway incrementally
- Old code removed only after new services verified

---

## Service Dependencies

```
gradio-ui → api-gateway → redis (queue)
                         → transcription (CUDA)
                         → translation (CUDA)
                         → ocr (CUDA optional)
                         → tts (CPU)
                         → subtitle (CPU)
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| UI | Gradio (separate service) |
| Task Queue | Celery + Redis |
| Message Broker | Redis |
| Result Backend | Redis |
| Container Orchestration | Docker Compose |
| Model Storage | Shared volume (`./models`) |

---

## Quick Start Guide for Contributors

1. **Set up development environment**:
   ```bash
   uv sync
   pre-commit install
   ```

2. **Start full stack**:
   ```bash
   docker-compose up -d
   ```

3. **Run tests**:
   ```bash
   pytest services/transcription/tests/
   ```

4. **Check code quality**:
   ```bash
   ruff check services/api-gateway/
   mypy services/api-gateway/
   ```

---

## Progress Tracking

- **Phase 0: Foundation**: 0/3 complete
- **Phase 1: Extract Services**: 0/5 complete
- **Phase 2: UI & Integration**: 0/3 complete
- **Phase 3: Quality & Production**: 0/5 complete
- **Phase 4: Optimization**: 0/4 complete
- **Total**: 0/20 tasks complete

---

*Last updated: 2026-04-13*
*Next review: After Phase 0 completion*
