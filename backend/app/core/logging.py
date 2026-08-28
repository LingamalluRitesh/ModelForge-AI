"""
ModelForge AI - Structured Logging & Telemetry Engine
Supports JSON-formatted structured logging for production log aggregators (ELK, Loki)
and colorized console logging for local developer workflows.
"""

import sys
import logging
import json
from datetime import datetime, timezone
from typing import Any, Dict
from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production observability."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "service": "modelforge-backend",
        }
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_obj["user_id"] = record.user_id
        if hasattr(record, "organization_id"):
            log_obj["organization_id"] = record.organization_id
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging() -> logging.Logger:
    """Initialize platform logging system based on active environment."""
    root_logger = logging.getLogger()
    root_logger.handlers = []

    level = logging.DEBUG if settings.DEBUG else logging.INFO
    root_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    if settings.ENVIRONMENT == "production":
        handler.setFormatter(JSONFormatter())
    else:
        color_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        handler.setFormatter(logging.Formatter(color_format))

    root_logger.addHandler(handler)

    # Silence overly verbose external loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("passlib").setLevel(logging.WARNING)

    return root_logger


logger = logging.getLogger("modelforge")
