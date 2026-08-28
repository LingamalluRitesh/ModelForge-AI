"""
ModelForge AI - Application Configuration
Comprehensive Pydantic Settings management for all services, databases,
queues, storage backends, and security protocols.
"""

from typing import List, Union, Optional
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # Application & Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PROJECT_NAME: str = "ModelForge AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "modelforge_super_secret_production_key_change_me_in_prod_at_least_32_chars"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    ALGORITHM: str = "HS256"

    # Server Host & Networking
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database Configuration (PostgreSQL with SQLite fallback for local test/dev)
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "modelforge_user"
    POSTGRES_PASSWORD: str = "modelforge_password"
    POSTGRES_DB: str = "modelforge_db"
    DATABASE_URL: Optional[str] = None
    DATABASE_URL_SYNC: Optional[str] = None
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_ECHO: bool = False

    @property
    def async_database_uri(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def sync_database_uri(self) -> str:
        if self.DATABASE_URL_SYNC:
            return self.DATABASE_URL_SYNC
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis Cache & Queue
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_URL: Optional[str] = None

    @property
    def redis_uri(self) -> str:
        if self.REDIS_URL:
            return self.REDIS_URL
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Celery Configuration
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    CELERY_TASK_ALWAYS_EAGER: bool = False

    @property
    def celery_broker(self) -> str:
        if self.CELERY_BROKER_URL:
            return self.CELERY_BROKER_URL
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/1"

    @property
    def celery_backend(self) -> str:
        if self.CELERY_RESULT_BACKEND:
            return self.CELERY_RESULT_BACKEND
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/2"

    # Storage Settings (Local Disk or S3 / MinIO)
    STORAGE_PROVIDER: str = "local"  # "local", "s3", "minio"
    STORAGE_LOCAL_PATH: str = str(BASE_DIR / "data_storage")
    S3_ENDPOINT_URL: Optional[str] = None
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "modelforge-artifacts"
    S3_REGION: str = "us-east-1"
    S3_SECURE: bool = False
    MAX_UPLOAD_SIZE_MB: int = 500

    # MLflow Tracking
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "ModelForge-Default"

    # Observability & Metrics
    PROMETHEUS_METRICS_ENABLED: bool = True
    OPENTELEMETRY_ENABLED: bool = False
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"

    # Email & Notifications (SMTP)
    SMTP_TLS: bool = True
    SMTP_PORT: int = 587
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: str = "alerts@modelforge.ai"
    EMAILS_FROM_NAME: str = "ModelForge AI Platform"

    # Security, Rate Limiting & Resilience
    RATE_LIMIT_PER_MINUTE: int = 120
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 30
    ALLOWED_HOSTS: List[str] = ["*"]

    # Initial Super Admin Seed
    FIRST_SUPERUSER_EMAIL: str = "admin@modelforge.ai"
    FIRST_SUPERUSER_PASSWORD: str = "AdminSecurePassword123!"
    FIRST_SUPERUSER_FIRSTNAME: str = "System"
    FIRST_SUPERUSER_LASTNAME: str = "Admin"
    FIRST_ORGANIZATION_NAME: str = "ModelForge Global Enterprise"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

# Ensure local storage directory exists
os.makedirs(settings.STORAGE_LOCAL_PATH, exist_ok=True)
os.makedirs(os.path.join(settings.STORAGE_LOCAL_PATH, "datasets"), exist_ok=True)
os.makedirs(os.path.join(settings.STORAGE_LOCAL_PATH, "models"), exist_ok=True)
os.makedirs(os.path.join(settings.STORAGE_LOCAL_PATH, "artifacts"), exist_ok=True)
os.makedirs(os.path.join(settings.STORAGE_LOCAL_PATH, "reports"), exist_ok=True)
