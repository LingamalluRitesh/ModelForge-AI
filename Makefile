# ModelForge AI — Platform Build & Orchestration Makefile
.PHONY: help install build run test clean docker-up docker-down

help:
	@echo "ModelForge AI — Commands:"
	@echo "  make install     - Install backend and frontend dependencies"
	@echo "  make build       - Build frontend static bundle and Docker containers"
	@echo "  make run         - Run the backend API server and Celery worker"
	@echo "  make test        - Run backend pytest test suites"
	@echo "  make docker-up   - Start full microservices stack via Docker Compose"
	@echo "  make docker-down - Teardown local containers"

install:
	pip install -r backend/requirements.txt
	cd frontend && npm install

build:
	cd frontend && npm run build
	docker build -t modelforge-api:latest -f infrastructure/docker/Dockerfile.api .
	docker build -t modelforge-frontend:latest -f infrastructure/docker/Dockerfile.frontend .

run:
	python run.py

test:
	cd backend && python -m pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf backend/.pytest_cache frontend/.next

docker-up:
	docker-compose -f infrastructure/docker-compose.yml up -d

docker-down:
	docker-compose -f infrastructure/docker-compose.yml down
