.PHONY: help dev up down logs test lint clean build shell

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development
dev: ## Start development environment
	docker-compose up -d

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

logs: ## View logs from all services
	docker-compose logs -f

logs-gateway: ## View API Gateway logs
	docker-compose logs -f api-gateway

logs-transcription: ## View Transcription service logs
	docker-compose logs -f transcription

logs-translation: ## View Translation service logs
	docker-compose logs -f translation

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

pull: ## Pull latest base images
	docker-compose pull

# Utilities
shell-gateway: ## Open shell in API Gateway container
	docker-compose exec api-gateway /bin/sh

shell-transcription: ## Open shell in Transcription container
	docker-compose exec transcription /bin/sh

shell-translation: ## Open shell in Translation container
	docker-compose exec translation /bin/sh

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
	@curl -s http://localhost:8000/health | jq .
	@curl -s http://localhost:8001/health | jq .
	@curl -s http://localhost:8002/health | jq .
	@curl -s http://localhost:8003/health | jq .
	@curl -s http://localhost:8004/health | jq .
	@curl -s http://localhost:8005/health | jq .
