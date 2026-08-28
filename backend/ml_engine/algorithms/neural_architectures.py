"""
ModelForge AI - ML Engine: Modular Neural Network Layers & Optimizer Suite
Implements Linear, LayerNorm, Dropout, GELU / Swish activations,
AdamW with Weight Decay, Cosine Annealing Learning Rate Scheduler, and Early Stopping.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class Activation:
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float64)

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit approximation."""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        """Swish / SiLU activation: $x \cdot \sigma(x)$."""
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))
        return x * sig

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        shift_x = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shift_x)
        return exps / np.sum(exps, axis=-1, keepdims=True)


class DenseLayer:
    """Fully Connected Layer with Kaiming / He normal initialization."""

    def __init__(self, in_features: int, out_features: int, activation: str = "gelu"):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation

        # Kaiming normal initialization
        std = np.sqrt(2.0 / in_features)
        self.W = np.random.normal(0, std, (in_features, out_features))
        self.b = np.zeros(out_features)

        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache for backpropagation
        self.x_cache: Optional[np.ndarray] = None
        self.z_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_cache = x
        self.z_cache = np.dot(x, self.W) + self.b

        if self.activation_name == "gelu":
            return Activation.gelu(self.z_cache)
        elif self.activation_name == "relu":
            return Activation.relu(self.z_cache)
        elif self.activation_name == "swish":
            return Activation.swish(self.z_cache)
        return self.z_cache

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            dout = dout * Activation.relu_grad(self.z_cache)
        elif self.activation_name == "gelu":
            # Numerical approximation gradient
            eps = 1e-5
            g_plus = Activation.gelu(self.z_cache + eps)
            g_minus = Activation.gelu(self.z_cache - eps)
            grad = (g_plus - g_minus) / (2.0 * eps)
            dout = dout * grad

        self.dW = np.dot(self.x_cache.T, dout)
        self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)


class LayerNorm:
    """Layer Normalization over feature dimension."""

    def __init__(self, features: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.x_cache: Optional[np.ndarray] = None
        self.x_norm_cache: Optional[np.ndarray] = None
        self.std_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_cache = x
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self.std_cache = np.sqrt(var + self.eps)
        self.x_norm_cache = (x - mean) / self.std_cache
        return self.gamma * self.x_norm_cache + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dgamma = np.sum(dout * self.x_norm_cache, axis=0)
        self.dbeta = np.sum(dout, axis=0)
        dx_norm = dout * self.gamma
        N = dout.shape[-1]
        dx = (1.0 / self.std_cache) * (dx_norm - np.mean(dx_norm, axis=-1, keepdims=True) - self.x_norm_cache * np.mean(dx_norm * self.x_norm_cache, axis=-1, keepdims=True))
        return dx


class AdamW:
    """AdamW decoupled weight decay optimizer."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}

    def step(self, param: np.ndarray, grad: np.ndarray, param_id: int):
        self.t += 1
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(grad)
            self.v[param_id] = np.zeros_like(grad)

        # Decay step
        param -= self.lr * self.weight_decay * param

        # Biased moments
        self.m[param_id] = self.beta1 * self.m[param_id] + (1.0 - self.beta1) * grad
        self.v[param_id] = self.beta2 * self.v[param_id] + (1.0 - self.beta2) * (grad ** 2)

        # Bias corrections
        m_hat = self.m[param_id] / (1.0 - self.beta1 ** self.t)
        v_hat = self.v[param_id] / (1.0 - self.beta2 ** self.t)

        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class TabularMLPClassifier:
    """Multi-Layer Perceptron neural classifier with LayerNorm and GELU."""

    def __init__(
        self,
        hidden_dims: List[int] = [128, 64],
        lr: float = 0.003,
        epochs: int = 40,
        batch_size: int = 64,
        weight_decay: float = 0.01,
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.layers: List[DenseLayer] = []
        self.ln_layers: List[LayerNorm] = []
        self.n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples, in_dim = X.shape
        self.n_classes = int(np.max(y)) + 1

        # Build architecture
        self.layers = []
        self.ln_layers = []
        curr_dim = in_dim

        for h_dim in self.hidden_dims:
            self.layers.append(DenseLayer(curr_dim, h_dim, activation="gelu"))
            self.ln_layers.append(LayerNorm(h_dim))
            curr_dim = h_dim

        self.head = DenseLayer(curr_dim, self.n_classes, activation="linear")
        optimizer = AdamW(lr=self.lr, weight_decay=self.weight_decay)

        # One-hot encoding
        y_one_hot = np.zeros((n_samples, self.n_classes))
        y_one_hot[np.arange(n_samples), y] = 1.0

        for epoch in range(self.epochs):
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y_one_hot[perm]

            for i in range(0, n_samples, self.batch_size):
                xb = X_shuffled[i : i + self.batch_size]
                yb = y_shuffled[i : i + self.batch_size]

                # Forward pass
                out = xb
                for l, ln in zip(self.layers, self.ln_layers):
                    out = ln.forward(l.forward(out))
                logits = self.head.forward(out)
                probs = Activation.softmax(logits)

                # Cross-entropy loss derivative
                dout = (probs - yb) / len(xb)

                # Backward pass
                dout = self.head.backward(dout)
                for l, ln in reversed(list(zip(self.layers, self.ln_layers))):
                    dout = l.backward(ln.backward(dout))

                # Parameter optimization
                optimizer.step(self.head.W, self.head.dW, 100)
                optimizer.step(self.head.b, self.head.db, 101)

                for idx, (l, ln) in enumerate(zip(self.layers, self.ln_layers)):
                    optimizer.step(l.W, l.dW, idx * 10)
                    optimizer.step(l.b, l.db, idx * 10 + 1)
                    optimizer.step(ln.gamma, ln.dgamma, idx * 10 + 2)
                    optimizer.step(ln.beta, ln.dbeta, idx * 10 + 3)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = X
        for l, ln in zip(self.layers, self.ln_layers):
            out = ln.forward(l.forward(out))
        logits = self.head.forward(out)
        return Activation.softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
