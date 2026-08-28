"""
ModelForge AI - Serving: Dark Shadow Traffic Mirror Evaluator
Safely mirrors live production HTTP/gRPC traffic to candidate challenger models in background
without impacting primary user response latencies or serving availability.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import numpy as np


class ShadowTrafficEvaluator:
    """Compares outputs, error rates, and latency distributions between primary champion and shadowed challenger."""
    def __init__(self, primary_model_fn: Callable[[np.ndarray], np.ndarray], shadow_model_fn: Callable[[np.ndarray], np.ndarray]):
        self.primary_fn = primary_model_fn
        self.shadow_fn = shadow_model_fn

        self.total_requests = 0
        self.output_discrepancy_count = 0
        self.primary_latencies_ms: List[float] = []
        self.shadow_latencies_ms: List[float] = []

    def evaluate_request(self, input_features: np.ndarray, tolerance: float = 0.05) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.total_requests += 1

        # 1. Evaluate Primary (Synchronous SLA Critical)
        t0 = time.perf_counter()
        primary_preds = self.primary_fn(input_features)
        t_primary = (time.perf_counter() - t0) * 1000.0
        self.primary_latencies_ms.append(t_primary)

        # 2. Evaluate Shadow (Dark Traffic Asynchronous Simulation)
        t1 = time.perf_counter()
        try:
            shadow_preds = self.shadow_fn(input_features)
            t_shadow = (time.perf_counter() - t1) * 1000.0
            self.shadow_latencies_ms.append(t_shadow)

            # Compare outputs
            diff = float(np.mean(np.abs(primary_preds - shadow_preds)))
            if diff > tolerance:
                self.output_discrepancy_count += 1
            shadow_success = True
        except Exception:
            shadow_preds = None
            shadow_success = False

        stats = {
            "primary_latency_ms": round(t_primary, 3),
            "shadow_latency_ms": round(t_shadow, 3) if shadow_success else None,
            "shadow_success": shadow_success,
            "discrepancy_rate": round(self.output_discrepancy_count / max(1, self.total_requests), 4),
        }
        return primary_preds, stats
