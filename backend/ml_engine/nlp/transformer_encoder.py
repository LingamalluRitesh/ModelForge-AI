"""
ModelForge AI - ML Engine: Bidirectional Transformer Encoder (BERT Architecture)
Implements Devlin et al. Bidirectional Encoder Representations from Transformers (BERT)
with Token & Position Embeddings, Multi-Head Attention, GELU Feed-Forward Networks, and Classification Head.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from ml_engine.nlp.multihead_attention import MultiHeadSelfAttention


class PositionWiseFeedForward:
    """Position-wise Two-Layer Feed-Forward Network: $FFN(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$."""

    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        self.d_model = d_model
        self.d_ff = d_ff

        std1 = np.sqrt(2.0 / d_model)
        self.W1 = np.random.normal(0, std1, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)

        std2 = np.sqrt(2.0 / d_ff)
        self.W2 = np.random.normal(0, std2, (d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self._gelu(np.dot(x, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        return out


class TransformerEncoderLayer:
    """Single Transformer Encoder Block with Pre-LayerNorm and Residual Connections."""

    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, eps: float = 1e-5):
        self.attn = MultiHeadSelfAttention(embed_dim=d_model, num_heads=num_heads, use_rope=False)
        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        self.eps = eps

    def _layernorm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * ((x - mean) / np.sqrt(var + self.eps)) + beta

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Self-attention sublayer with residual
        norm_x1 = self._layernorm(x, self.gamma1, self.beta1)
        attn_out, _, _ = self.attn.forward(norm_x1, is_causal=False)
        x = x + attn_out

        # FFN sublayer with residual
        norm_x2 = self._layernorm(x, self.gamma2, self.beta2)
        ffn_out = self.ffn.forward(norm_x2)
        x = x + ffn_out

        return x


class TransformerEncoder:
    """Full Bidirectional Transformer Encoder Stack."""

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        num_classes: int = 2,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes

        # Embeddings
        self.token_embeddings = np.random.normal(0, 0.02, (vocab_size, d_model))
        self.position_embeddings = np.random.normal(0, 0.02, (max_seq_len, d_model))

        # Layers
        self.layers = [
            TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
            for _ in range(num_layers)
        ]

        # Classification Head (Pooler)
        self.pooler_W = np.random.normal(0, 0.02, (d_model, d_model))
        self.pooler_b = np.zeros(d_model)
        self.classifier_W = np.random.normal(0, 0.02, (d_model, num_classes))
        self.classifier_b = np.zeros(num_classes)

    def forward(self, input_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass producing contextual sequence tokens and pooled sentence classification logits.
        input_ids shape: $(B, S)$
        """
        B, S = input_ids.shape
        positions = np.arange(S)

        # Sum token and position embeddings
        x = self.token_embeddings[input_ids] + self.position_embeddings[positions]

        # Pass through sequential encoder layers
        for layer in self.layers:
            x = layer.forward(x)

        # Pool [CLS] token at position 0
        cls_token = x[:, 0, :]
        pooled = np.tanh(np.dot(cls_token, self.pooler_W) + self.pooler_b)
        logits = np.dot(pooled, self.classifier_W) + self.classifier_b

        return x, logits
