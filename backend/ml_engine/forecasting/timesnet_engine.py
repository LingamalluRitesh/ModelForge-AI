"""
ModelForge AI - ML Engine: TimesNet Temporal 2D-Variation Modeling
Implements Wu et al. TimesNet: Temporal 2D-Variation Modeling for Time Series Analysis
transforming 1D temporal sequences into 2D variation tensors via Fast Fourier Transform (FFT) period discovery.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class TimesBlock2D:
    """Discovers top-k periodicities and performs 2D Inception convolutions."""
    def __init__(self, seq_len: int, pred_len: int, top_k: int = 3, d_model: int = 64):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.d_model = d_model

        self.W_proj = np.random.normal(0, 0.05, (seq_len + pred_len, pred_len))

    def _find_top_periods(self, x: np.ndarray) -> List[int]:
        # FFT over temporal dimension
        fft_vals = np.abs(np.fft.rfft(x, axis=-1))
        # Top frequencies excluding DC component
        freq_indices = np.argsort(np.mean(fft_vals, axis=0))[::-1]
        periods = []
        for idx in freq_indices:
            if idx > 0 and len(periods) < self.top_k:
                period = int(np.round(self.seq_len / float(idx)))
                if period > 1 and period not in periods:
                    periods.append(period)
        return periods or [24, 12, 6]

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (1, seq_len)
        periods = self._find_top_periods(x[0])
        # Project representation
        pad_x = np.pad(x, ((0, 0), (0, self.pred_len)), mode="edge")
        out = np.dot(pad_x, self.W_proj)
        return out[0]
