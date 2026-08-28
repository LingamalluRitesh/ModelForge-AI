"""
ModelForge AI - Common Schemas & Envelope Types
"""

from typing import Any, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standardized API response wrapper."""
    success: bool = True
    data: Optional[T] = None
    message: Optional[str] = None
    request_id: Optional[str] = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Standardized pagination response envelope."""
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


class SuccessResponse(BaseModel):
    success: bool = True
    message: str = "Operation completed successfully."
