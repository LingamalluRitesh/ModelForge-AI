"""
ModelForge AI - Preprocessing: Fourier Harmonic Seasonality & Cyclical Feature Encoder
Transforms timestamps into trigonometric sine and cosine basis expansions: $\sin\left(rac{2\pi k t}{P}ight), \cos\left(rac{2\pi k t}{P}ight)$.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class CyclicalTimeEncoder:
    """Encodes calendar variables into continuous cyclical representations."""
    def __init__(self, period_days: float = 365.25, num_harmonics: int = 4):
        self.period = period_days
        self.K = num_harmonics

    def transform(self, time_in_days: np.ndarray) -> np.ndarray:
        t = np.asarray(time_in_days, dtype=float)
        harmonics = []

        for k in range(1, self.K + 1):
            angle = (2.0 * np.pi * k * t) / self.period
            harmonics.append(np.sin(angle))
            harmonics.append(np.cos(angle))

        return np.column_stack(harmonics)
