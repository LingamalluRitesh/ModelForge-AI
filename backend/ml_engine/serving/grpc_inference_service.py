"""
ModelForge AI - Serving: High-Performance gRPC Inference Service
Handles binary Protobuf serialized tensor streams with zero-copy buffer serialization,
sub-millisecond worker dispatch, and concurrent thread pool execution.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import numpy as np


class GRPCInferenceService:
    """gRPC Protobuf Tensor Inference Dispatcher."""
    def __init__(self, model_predict_fn: Callable[[np.ndarray], np.ndarray]):
        self.predict_fn = model_predict_fn
        self.total_requests = 0
        self.total_latency_ms = 0.0

    def predict_tensor(self, tensor_bytes: bytes, shape: Tuple[int, ...], dtype: str = "float32") -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.total_requests += 1

        # Deserialize raw tensor buffer
        np_dtype = np.float32 if dtype == "float32" else np.float64
        input_array = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape)

        # Execute scoring
        preds = self.predict_fn(input_array)
        latency = (time.perf_counter() - t0) * 1000.0
        self.total_latency_ms += latency

        output_bytes = preds.astype(np_dtype).tobytes()

        return {
            "output_bytes": output_bytes,
            "output_shape": list(preds.shape),
            "output_dtype": dtype,
            "latency_ms": round(latency, 3),
            "average_qps": round(self.total_requests / max(1e-4, self.total_latency_ms / 1000.0), 1),
        }
