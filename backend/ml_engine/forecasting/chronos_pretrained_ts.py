"""
ModelForge AI - Forecasting Engine: Chronos Tokenized Time Series Language Model
Implements Ansari et al. Chronos: Learning the Language of Time Series
tokenizing continuous 1D time series into quantized discrete token vocabularies and predicting autoregressively.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ChronosTokenizer:
    """Quantizes continuous values into discrete vocabulary tokens via Mean-Scaling and Uniform Binning."""
    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.bins = np.linspace(-15.0, 15.0, vocab_size - 2)

    def encode(self, series: np.ndarray) -> Tuple[np.ndarray, float]:
        """Returns token IDs and mean scale factor."""
        x = np.asarray(series, dtype=float)
        scale = float(np.mean(np.abs(x))) + 1e-5
        norm_x = x / scale
        token_ids = np.digitize(norm_x, self.bins)
        return token_ids, scale

    def decode(self, token_ids: np.ndarray, scale: float) -> np.ndarray:
        """Converts token IDs back to continuous time series."""
        bin_centers = np.concatenate([[-15.0], (self.bins[:-1] + self.bins[1:]) / 2.0, [15.0]])
        norm_x = bin_centers[np.clip(token_ids, 0, len(bin_centers) - 1)]
        return norm_x * scale


class ChronosForecaster:
    """Chronos Autoregressive Transformer Time Series Forecaster."""
    def __init__(self, vocab_size: int = 4096, d_model: int = 128):
        self.tokenizer = ChronosTokenizer(vocab_size)
        self.W_emb = np.random.normal(0, 0.05, (vocab_size, d_model))
        self.W_head = np.random.normal(0, 0.05, (d_model, vocab_size))

    def forecast(self, history: np.ndarray, forecast_steps: int = 12) -> np.ndarray:
        tokens, scale = self.tokenizer.encode(history)
        generated_tokens = list(tokens)

        for _ in range(forecast_steps):
            last_tok = generated_tokens[-1]
            emb = self.W_emb[last_tok]
            logits = np.dot(emb, self.W_head)
            next_tok = int(np.argmax(logits))
            generated_tokens.append(next_tok)

        pred_tokens = np.array(generated_tokens[-forecast_steps:])
        return self.tokenizer.decode(pred_tokens, scale)
