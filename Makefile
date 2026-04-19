.PHONY: help dev up down logs test lint clean build shell

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development
dev: ## Start development environment (all services)
	docker-compose --profile all up -d

up: ## Start all services
	docker-compose --profile all up -d

up-minimal: ## Start minimal (UI + Gateway + Redis)
	docker-compose --profile minimal up -d

up-tts: ## Start only TTS service
	docker-compose --profile tts-only up -d

up-transcription: ## Start only Transcription service
	docker-compose --profile transcription-only up -d

up-translation: ## Start only Translation service
	docker-compose --profile translation-only up -d

down: ## Stop all services
	docker-compose --profile all down

logs: ## View logs from all services
	docker-compose logs -f

logs-gateway: ## View API Gateway logs
	docker-compose logs -f api-gateway

logs-transcription: ## View Transcription service logs
	docker-compose logs -f transcription

logs-translation: ## View Translation service logs
	docker-compose logs -f translation

logs-ocr: ## View OCR service logs
	docker-compose logs -f ocr

logs-tts: ## View TTS service logs
	docker-compose logs -f tts

logs-subtitle: ## View Subtitle service logs
	docker-compose logs -f subtitle

logs-gradio: ## View Gradio UI logs
	docker-compose logs -f gradio-ui

# Testing
test: ## Run tests
	pytest services/ -v

test-transcription: ## Run transcription tests
	pytest services/transcription/tests/ -v

test-translation: ## Run translation tests
	pytest services/translation/tests/ -v

test-ocr: ## Run OCR tests
	pytest services/ocr/tests/ -v

test-tts: ## Run TTS tests
	pytest services/tts/tests/ -v

test-subtitle: ## Run subtitle tests
	pytest services/subtitle/tests/ -v

test-coverage: ## Run tests with coverage
	pytest services/ --cov=services --cov-report=html

# Code Quality
lint: ## Run linter on all services
	ruff check services/

lint-fix: ## Fix linting issues
	ruff check --fix services/

format: ## Format code
	ruff format services/

type-check: ## Run type checker
	mypy services/

# Docker
build: ## Build all Docker images
	docker-compose build

build-transcription: ## Build transcription service image
	docker-compose build transcription

build-translation: ## Build translation service image
	docker-compose build translation

build-ocr: ## Build OCR service image
	docker-compose build ocr

build-tts: ## Build TTS service image
	docker-compose build tts

build-subtitle: ## Build subtitle service image
	docker-compose build subtitle

build-gradio: ## Build Gradio UI image
	docker-compose build gradio-ui

build-gateway: ## Build API Gateway image
	docker-compose build api-gateway

pull: ## Pull latest base images
	docker-compose pull

# Utilities
shell-gateway: ## Open shell in API Gateway container
	docker-compose exec api-gateway /bin/sh

shell-transcription: ## Open shell in Transcription container
	docker-compose exec transcription /bin/sh

shell-translation: ## Open shell in Translation container
	docker-compose exec translation /bin/sh

shell-ocr: ## Open shell in OCR container
	docker-compose exec ocr /bin/sh

shell-tts: ## Open shell in TTS container
	docker-compose exec tts /bin/sh

shell-subtitle: ## Open shell in Subtitle container
	docker-compose exec subtitle /bin/sh

shell-gradio: ## Open shell in Gradio container
	docker-compose exec gradio-ui /bin/sh

redis-cli: ## Open Redis CLI
	docker-compose exec redis redis-cli

redis-monitor: ## Open Redis Commander (debug)
	docker-compose --profile debug up -d redis-commander

clean: ## Clean up Docker resources
	docker-compose down -v --remove-orphans
	docker system prune -f

restart: ## Restart all services
	docker-compose restart

health: ## Check health of all services
	@echo "Checking service health..."
	@echo "API Gateway: "
	@curl -s http://localhost:8000/health || echo "not running"
	@echo "Transcription: "
	@curl -s http://localhost:8001/health || echo "not running"
	@echo "Translation: "
	@curl -s http://localhost:8002/health || echo "not running"
	@echo "OCR: "
	@curl -s http://localhost:8003/health || echo "not running"
	@echo "TTS: "
	@curl -s http://localhost:8004/health || echo "not running"
	@echo "Subtitle: "
	@curl -s http://localhost:8005/health || echo "not running"
	@echo "Gradio UI: "
	@curl -s http://localhost:7860 || echo "not running"
