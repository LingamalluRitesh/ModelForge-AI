"""
Prometheus Metrics REST Endpoints.
"""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from typing import Dict, Any

from app.services.gpu_memory_profiler import GpuMemoryProfiler

router = APIRouter(prefix="/metrics", tags=["Observability & Metrics"])
profiler = GpuMemoryProfiler()


@router.get("/summary", summary="Get GPU Memory Summary JSON")
def get_summary() -> Dict[str, Any]:
    return profiler.get_cluster_memory_summary()


@router.get("/prometheus", summary="Expose Prometheus Formatted Metrics", response_class=PlainTextResponse)
def get_prometheus_metrics() -> str:
    return profiler.export_prometheus_metrics()
