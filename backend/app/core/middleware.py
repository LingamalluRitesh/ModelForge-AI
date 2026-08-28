"""
ModelForge AI - FastAPI Middleware Suite
Request IDs, structured timing, security headers, CORS, audit hooks, and performance tracking.
"""

import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import logger


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Assigns unique request_id to each incoming request, records execution latency,
    and sets standard enterprise security response headers.
    """

    async def dispatch(self, request: Request, call_next):
        # Extract or generate Request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        start_time = time.time()

        try:
            response: Response = await call_next(request)
        except Exception as e:
            process_time = (time.time() - start_time) * 1000
            logger.error(
                f"Request Failed: {request.method} {request.url.path} "
                f"[{request_id}] - Latency: {process_time:.2f}ms - Error: {str(e)}"
            )
            raise e

        process_time = (time.time() - start_time) * 1000

        # Inject standard security and telemetry response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{process_time:.2f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Log request at appropriate severity
        if response.status_code >= 500:
            logger.error(
                f"{request.method} {request.url.path} {response.status_code} "
                f"[{request_id}] - {process_time:.2f}ms"
            )
        elif response.status_code >= 400:
            logger.warning(
                f"{request.method} {request.url.path} {response.status_code} "
                f"[{request_id}] - {process_time:.2f}ms"
            )
        else:
            logger.info(
                f"{request.method} {request.url.path} {response.status_code} "
                f"[{request_id}] - {process_time:.2f}ms"
            )

        return response
