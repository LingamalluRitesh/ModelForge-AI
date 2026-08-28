"""
ModelForge AI - ML Engine: Asynchronous Distributed Parameter Server
Implements Li et al. Scaling Distributed Machine Learning with Parameter Server Architecture
supporting asynchronous gradient accumulation, Staleness-Aware Learning Rate decay, and bounded delay.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import threading
import time
import numpy as np


class ParameterServer:
    """Central Parameter Server coordinating decentralized worker gradient updates."""

    def __init__(self, param_dim: int, lr: float = 0.01, max_staleness: int = 4):
        self.param_dim = param_dim
        self.lr = lr
        self.max_staleness = max_staleness

        self.weights = np.zeros(param_dim, dtype=np.float64)
        self.global_clock = 0
        self.lock = threading.Lock()
        self.worker_versions: Dict[str, int] = {}

    def pull_weights(self, worker_id: str) -> Tuple[np.ndarray, int]:
        """Worker pulls latest weights along with current global clock version."""
        with self.lock:
            self.worker_versions[worker_id] = self.global_clock
            return self.weights.copy(), self.global_clock

    def push_gradients(self, worker_id: str, gradients: np.ndarray, worker_version: int) -> bool:
        """
        Worker pushes gradients calculated against worker_version weights.
        Applies staleness penalty $\eta_{eff} = \frac{\eta}{1 + \tau}$.
        """
        with self.lock:
            staleness = self.global_clock - worker_version

            if staleness > self.max_staleness:
                # Discard overly stale gradient update
                return False

            # Staleness-damped learning rate
            effective_lr = self.lr / (1.0 + 0.5 * staleness)

            # Gradient update
            self.weights -= effective_lr * np.asarray(gradients, dtype=np.float64)
            self.global_clock += 1
            return True
