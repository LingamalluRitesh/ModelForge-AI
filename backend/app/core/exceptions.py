"""
ModelForge AI - Centralized Exception Hierarchy & Handlers
Standardized error formatting with error codes, request IDs, and safe messages.
"""

from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse


class ModelForgeException(Exception):
    """Base exception for all ModelForge AI platform errors."""
    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_SERVER_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class EntityNotFoundException(ModelForgeException):
    def __init__(self, entity_name: str, entity_id: Any, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"{entity_name} with ID '{entity_id}' was not found.",
            code=f"{entity_name.upper()}_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class EntityAlreadyExistsException(ModelForgeException):
    def __init__(self, entity_name: str, identifier: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"{entity_name} with identifier '{identifier}' already exists.",
            code=f"{entity_name.upper()}_ALREADY_EXISTS",
            status_code=status.HTTP_409_CONFLICT,
            details=details,
        )


class AuthenticationException(ModelForgeException):
    def __init__(self, message: str = "Authentication failed or token invalid.", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="AUTHENTICATION_FAILED",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
        )


class AuthorizationException(ModelForgeException):
    def __init__(self, message: str = "You do not have permission to perform this action.", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="PERMISSION_DENIED",
            status_code=status.HTTP_403_FORBIDDEN,
            details=details,
        )


class ValidationException(ModelForgeException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


class RateLimitExceededException(ModelForgeException):
    def __init__(self, message: str = "API rate limit exceeded. Please retry later.", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
        )


class CircuitBreakerOpenException(ModelForgeException):
    def __init__(self, service_name: str):
        super().__init__(
            message=f"Service '{service_name}' is currently unavailable due to circuit breaker trip.",
            code="CIRCUIT_BREAKER_OPEN",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class MLModelExecutionException(ModelForgeException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="ML_EXECUTION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class DriftDetectionException(ModelForgeException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="DRIFT_DETECTION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


async def modelforge_exception_handler(request: Request, exc: ModelForgeException) -> JSONResponse:
    """Standardized JSON error response handler for ModelForge exceptions."""
    request_id = getattr(request.state, "request_id", "unknown-request-id")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "request_id": request_id,
            },
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Centralized fallback exception handler for uncaught server exceptions."""
    request_id = getattr(request.state, "request_id", "unknown-request-id")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected internal server error occurred. Please contact support.",
                "details": {"error_type": exc.__class__.__name__},
                "request_id": request_id,
            },
        },
    )
