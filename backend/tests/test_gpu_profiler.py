import pytest
from backend.app.services.gpu_memory_profiler import GpuMemoryProfiler


def test_gpu_profiler_allocation_and_metrics():
    profiler = GpuMemoryProfiler(device_count=2)
    summary = profiler.get_cluster_memory_summary()
    assert summary["total_devices"] == 2
    assert summary["total_capacity_vram_mb"] == 49152.0

    profiler.record_allocation(device_id=0, vram_mb=512.0)
    new_summary = profiler.get_cluster_memory_summary()
    assert new_summary["devices"][0]["allocated_vram_mb"] == 2560.0

    prom_text = profiler.export_prometheus_metrics()
    assert 'modelforge_gpu_allocated_vram_mb{device="gpu_0"}' in prom_text
    assert 'modelforge_inferences_total{device="gpu_0"}' in prom_text
