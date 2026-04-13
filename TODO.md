# TODO: Local Media Processor - Microservices Migration

This file tracks the migration from monolithic app.py to microservices architecture.
Tasks are organized in phases for sequential implementation.

---

## Phase 0: Foundation (Week 1-2)

### 1. Create Project Structure
**Problem**: Single-file monolith needs to become multi-service architecture
**Status**: ✅ COMPLETE
**Steps**:
- [x] Create root workspace structure
- [x] Create `services/shared/` package with common types, utils, config
- [x] Set up individual service directories with standard FastAPI structure
- [x] Create base Dockerfile template for services
- [x] Update root docker-compose.yml for multi-service setup

### 2. Set Up Redis + Celery Infrastructure
**Files**: `docker-compose.yml`, shared Celery config
**Status**: ✅ COMPLETE
**Steps**:
- [x] Add Redis service to docker-compose.yml
- [x] Add Redis configuration with persistence
- [x] Create shared Celery app configuration in `services/shared/celery_app.py`
- [x] Create base Celery worker template
- [x] Configure Redis as message broker and result backend
- [x] Test Redis connectivity from services
- [x] Add Redis monitoring/health check

### 3. Create API Gateway Service
**Directory**: `services/api-gateway/`
**Framework**: FastAPI
**Status**: ✅ COMPLETE
**Steps**:
- [x] Create FastAPI application with basic routing
- [x] Implement authentication middleware (JWT or API key)
- [x] Add rate limiting middleware
- [x] Create route table (all endpoints defined)
- [x] Add request validation and sanitization
- [x] Implement centralized error handling
- [x] Add OpenAPI/Swagger docs
- [x] Add health check endpoint `/health`
- [x] Create Dockerfile for API gateway
- [x] Test gateway routes with mock services
- [ ] Test gateway routes with real services (Phase 1)

---

## Phase 1: Extract Core Services (Week 3-5)

### 4. Extract Transcription Service
**Directory**: `services/transcription/`
**Current**: `app.py` transcription logic
**Status**: 🟡 PARTIAL (Structure, Dockerfile, Celery worker created)
**Steps**:
- [x] Create FastAPI service with `/transcribe` endpoint
- [ ] Move faster-whisper logic from app.py
- [ ] Move ModelManager (whisper part) to service
- [x] Create Celery task for long transcription jobs
- [ ] Implement async task submission (return task_id)
- [ ] Add task status endpoint
- [ ] Add file upload handling (multipart or path)
- [x] Add GPU configuration
- [ ] Create service-specific config
- [ ] Remove bare except clauses
- [ ] Add proper error handling and logging
- [x] Create Dockerfile (CUDA base image)
- [ ] Add service tests
- [ ] Test with API gateway integration

### 5. Extract Translation Service
**Directory**: `services/translation/`
**Current**: `app.py` translation logic
**Status**: 🟡 PARTIAL (Structure, Dockerfile, Celery worker created)
**Steps**:
- [x] Create FastAPI service with `/translate` endpoint
- [ ] Move NLLB translation logic from app.py
- [ ] Move ModelManager (NLLB part) to service
- [x] Create Celery task for translation jobs
- [ ] Implement chunked translation (fix [:3000] truncation)
- [x] Add language code validation
- [x] Add GPU/CPU configuration
- [ ] Create service-specific config
- [ ] Remove bare except clauses
- [ ] Add proper error handling
- [x] Create Dockerfile (CUDA base image)
- [ ] Add service tests
- [ ] Test with API gateway integration

### 6. Extract OCR Service
**Directory**: `services/ocr/`
**Current**: `app.py` OCR logic
**Status**: 🟡 PARTIAL (Structure, Dockerfile, Celery worker created)
**Steps**:
- [x] Create FastAPI service with `/ocr` endpoint
- [ ] Move EasyOCR logic from app.py
- [ ] Move ModelManager (OCR part) to service
- [x] Create Celery task for OCR jobs
- [ ] Add image format validation
- [x] Add GPU/CPU configuration
- [ ] Create service-specific config
- [ ] Remove bare except clauses
- [ ] Add proper error handling
- [x] Create Dockerfile (CUDA or CPU image)
- [ ] Add service tests
- [ ] Test with API gateway integration

### 7. Extract TTS Service
**Directory**: `services/tts/`
**Current**: `app.py` TTS logic
**Status**: 🟡 PARTIAL (Structure, Dockerfile, Celery worker created)
**Steps**:
- [x] Create FastAPI service with `/tts` endpoint
- [ ] Move edge_tts logic from app.py
- [x] Create Celery task for TTS jobs
- [ ] Add voice selection validation
- [ ] Add language-to-voice mapping
- [ ] Create service-specific config
- [ ] Remove bare except clauses
- [ ] Add proper error handling
- [x] Create Dockerfile (CPU-only, no CUDA needed)
- [ ] Add service tests
- [ ] Test with API gateway integration

### 8. Extract Subtitle Service
**Directory**: `services/subtitle/`
**Current**: `app.py` SRT generation logic
**Status**: 🟡 PARTIAL (Structure, Dockerfile, Celery worker created)
**Steps**:
- [x] Create FastAPI service with `/subtitle` endpoint
- [ ] Move pysrt logic from app.py
- [x] Create Celery task for subtitle generation
- [ ] Add SRT format validation
- [ ] Create service-specific config
- [ ] Add proper error handling
- [x] Create Dockerfile (lightweight, no GPU)
- [ ] Add service tests
- [ ] Test with API gateway integration

---

## Phase 2: UI and Integration (Week 6-7)

### 9. Refactor Gradio UI Service
**Directory**: `services/gradio-ui/`
**Current**: `app.py` UI logic
**Status**: 🟡 PARTIAL (Structure, placeholder UI created)
**Steps**:
- [x] Create separate FastAPI + Gradio service
- [ ] Move Gradio interface from app.py
- [ ] Update UI to call API gateway instead of direct processing
- [ ] Implement async task polling (show progress)
- [ ] Add task cancellation support in UI
- [ ] Add batch upload with progress tracking
- [ ] Add auth configuration (env var GRADIO_AUTH)
- [ ] Update file input to work with API gateway
- [ ] Remove all processing logic from UI service
- [x] Create Dockerfile
- [ ] Test complete UI flow with gateway
- [ ] Test batch processing through new architecture

### 10. Implement Shared Components
**Directory**: `services/shared/`
**Status**: ✅ COMPLETE
**Steps**:
- [x] Create shared configuration module
- [x] Create shared Celery app factory
- [x] Create shared logging configuration
- [x] Create shared file I/O utilities
- [x] Create shared language codes/validation
- [x] Create shared Pydantic models for requests/responses
- [x] Create shared error types
- [x] Add shared test utilities
- [x] Package as installable Python package
- [ ] Add shared package tests

### 11. Create Docker Compose Setup
**File**: `docker-compose.yml`
**Status**: ✅ COMPLETE
**Steps**:
- [x] Define all services
- [x] Configure shared networks
- [x] Configure volume mounts
- [x] Add service dependencies (depends_on)
- [x] Add health checks for all services
- [x] Add resource limits (memory, GPU)
- [x] Create development and production compose files
- [ ] Test full stack startup
- [ ] Document docker-compose usage in README

---

## Phase 3: Quality and Production (Week 8-9)

### 12. Add Authentication & Security
**Status**: 🟡 PARTIAL (Non-root users configured)
**Steps**:
- [ ] Implement JWT or API key auth in API gateway
- [ ] Add auth middleware to protect endpoints
- [ ] Configure Gradio auth for UI access
- [ ] Add path validation and sanitization in all services
- [x] Run all Docker containers as non-root users
- [ ] Add checksum verification for model downloads
- [ ] Add rate limiting in API gateway
- [ ] Add request size limits
- [ ] Test auth flows end-to-end

### 13. Add Monitoring & Observability
**Status**: 🟡 PARTIAL (Health checks, logging configured)
**Steps**:
- [ ] Add structured logging (JSON) to all services
- [ ] Add request ID tracing across services
- [x] Add health check endpoints to all services
- [ ] Add metrics collection (Prometheus or similar)
- [ ] Add task monitoring (Celery flower or custom dashboard)
- [ ] Add error tracking (Sentry or similar)
- [ ] Create monitoring dashboard
- [ ] Add log aggregation
- [ ] Document monitoring setup

### 14. Add Testing Infrastructure
**Directory**: `tests/` in each service
**Status**: 🟡 PARTIAL (Test directories created, infrastructure test exists)
**Steps**:
- [x] Set up pytest in each service
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
- [x] Create Makefile with common commands
- [x] Add .env.example files
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

- **Phase 0: Foundation**: 3/3 complete (✅ Project Structure, ✅ Redis/Celery, ✅ API Gateway)
- **Phase 1: Extract Services**: 0/5 complete (🟡 All 5 services have structure + Dockerfiles + Celery workers)
- **Phase 2: UI & Integration**: 2/3 complete (🟡 UI, ✅ Shared Components, ✅ Docker Compose)
- **Phase 3: Quality & Production**: 0/5 complete (🟡 Partial progress on auth, monitoring, testing)
- **Phase 4: Optimization**: 0/4 complete (🟡 Partial: Makefile, .env.example created)
- **Total**: 5/20 tasks fully complete, 12 partially complete, 3 not started

### Completed Items:
✅ Project workspace structure  
✅ Redis + Celery infrastructure  
✅ Shared components package (installable, tested)  
✅ Docker Compose multi-service setup  
✅ All service Dockerfiles  
✅ All service pyproject.toml files  
✅ Celery worker templates for all services  
✅ Health checks for all services  
✅ Non-root Docker containers  
✅ Makefile with dev commands  
✅ .env.example configuration  
✅ Infrastructure test script  
✅ .dockerignore file  
✅ Service README files  
✅ **API Gateway service** (routes, task management, auth, rate limiting)  
✅ **API Gateway tests** (7/7 passing)  

### Next Priority Items:
🔴 Implement actual service logic (move from app.py)  
🔴 Connect API Gateway to services  
🔴 Write unit tests for services  
🟡 Test full stack with Docker Compose    

---

*Last updated: 2026-04-13*
*Next review: After Phase 0.3 (API Gateway) completion*
