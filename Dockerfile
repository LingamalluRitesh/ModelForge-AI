# ModelForge AI — Root Multi-Stage Production Dockerfile
FROM python:3.12-slim AS backend-builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ ./backend/
COPY sdk/ ./sdk/
COPY run.py app.py Makefile ./

EXPOSE 8000
CMD ["python", "run.py"]
