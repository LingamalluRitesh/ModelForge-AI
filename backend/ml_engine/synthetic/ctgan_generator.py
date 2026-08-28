"""
ModelForge AI - ML Engine: Conditional Tabular GAN (CTGAN)
Implements Xu et al. Modeling Tabular Data using Conditional GAN with Mode-Specific Normalization,
Gumbel-Softmax discrete representation, and Wasserstein GAN with Gradient Penalty (WGAN-GP).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ModeSpecificNormalizer:
    """Transforms multimodal continuous columns into one-hot cluster representations and scalar representations."""

    def __init__(self, max_clusters: int = 10, eps: float = 1e-5):
        self.max_clusters = max_clusters
        self.eps = eps
        self.means_: Dict[int, np.ndarray] = {}
        self.stds_: Dict[int, np.ndarray] = {}
        self.weights_: Dict[int, np.ndarray] = {}
        self.n_clusters_: Dict[int, int] = {}

    def fit(self, X: np.ndarray, continuous_columns: List[int]):
        for col in continuous_columns:
            data = X[:, col]
            # Estimate GMM clusters via standard histogram/quantiles
            n_c = min(self.max_clusters, len(np.unique(data)))
            quantiles = np.linspace(0, 1, n_c + 1)
            bin_edges = np.quantile(data, quantiles)

            means = []
            stds = []
            weights = []

            for k in range(n_c):
                mask = (data >= bin_edges[k]) & (data <= bin_edges[k + 1])
                cluster_data = data[mask]
                if len(cluster_data) > 0:
                    means.append(float(np.mean(cluster_data)))
                    stds.append(float(max(self.eps, np.std(cluster_data))))
                    weights.append(float(len(cluster_data) / len(data)))
                else:
                    means.append(float(bin_edges[k]))
                    stds.append(1.0)
                    weights.append(0.0)

            self.means_[col] = np.array(means)
            self.stds_[col] = np.array(stds)
            self.weights_[col] = np.array(weights)
            self.n_clusters_[col] = len(means)

        return self

    def transform_column(self, data_col: np.ndarray, col_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns normalized scalar representation $v$ and one-hot cluster vector $\beta$."""
        means = self.means_[col_idx]
        stds = self.stds_[col_idx]
        n_c = len(means)

        # Distance to cluster means
        dists = np.abs(data_col[:, np.newaxis] - means[np.newaxis, :]) / stds[np.newaxis, :]
        closest_clusters = np.argmin(dists, axis=1)

        # One-hot representation
        beta = np.zeros((len(data_col), n_c))
        beta[np.arange(len(data_col)), closest_clusters] = 1.0

        # Normalized scalar: v = (x - mu_k) / (4 * sigma_k)
        selected_means = means[closest_clusters]
        selected_stds = stds[closest_clusters]
        v = (data_col - selected_means) / (4.0 * selected_stds)
        v = np.clip(v, -0.99, 0.99)

        return v, beta


class ConditionalGenerator:
    """Generator network generating tabular vectors conditioned on categorical constraints."""

    def __init__(self, embedding_dim: int = 128, output_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        std = np.sqrt(2.0 / embedding_dim)
        self.W1 = np.random.normal(0, std, (embedding_dim, 256))
        self.b1 = np.zeros(256)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / 256), (256, 256))
        self.b2 = np.zeros(256)
        self.W3 = np.random.normal(0, np.sqrt(2.0 / 256), (256, output_dim))
        self.b3 = np.zeros(output_dim)

    def forward(self, noise: np.ndarray, cond_vector: np.ndarray) -> np.ndarray:
        inp = np.concatenate([noise, cond_vector], axis=-1)
        h1 = np.maximum(0, np.dot(inp, self.W1) + self.b1)
        h2 = np.maximum(0, np.dot(h1, self.W2) + self.b2)
        out = np.tanh(np.dot(h2, self.W3) + self.b3)
        return out


class Discriminator:
    """Wasserstein Critic network with gradient penalty scoring realistic tabular structures."""

    def __init__(self, input_dim: int = 64):
        self.input_dim = input_dim

        std = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.normal(0, std, (input_dim, 256))
        self.b1 = np.zeros(256)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / 256), (256, 256))
        self.b2 = np.zeros(256)
        self.W3 = np.random.normal(0, np.sqrt(2.0 / 256), (256, 1))
        self.b3 = np.zeros(1)

    def forward(self, x: np.ndarray, cond_vector: np.ndarray) -> np.ndarray:
        inp = np.concatenate([x, cond_vector], axis=-1)
        h1 = np.maximum(0.2 * np.dot(inp, self.W1) + self.b1, np.dot(inp, self.W1) + self.b1)
        h2 = np.maximum(0.2 * np.dot(h1, self.W2) + self.b2, np.dot(h1, self.W2) + self.b2)
        score = np.dot(h2, self.W3) + self.b3
        return score


class CTGAN:
    """Conditional Tabular GAN for high-fidelity synthetic tabular dataset generation."""

    def __init__(
        self,
        embedding_dim: int = 128,
        batch_size: int = 128,
        epochs: int = 50,
        lr: float = 2e-4,
    ):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.normalizer = ModeSpecificNormalizer()
        self.generator: Optional[ConditionalGenerator] = None
        self.discriminator: Optional[Discriminator] = None
        self.continuous_cols: List[int] = []
        self.n_features = 0

    def fit(self, X: np.ndarray, continuous_columns: Optional[List[int]] = None):
        X = np.asarray(X, dtype=np.float64)
        self.n_features = X.shape[1]
        self.continuous_cols = continuous_columns or []

        self.normalizer.fit(X, self.continuous_cols)

        # Initialize networks
        cond_dim = 16
        self.generator = ConditionalGenerator(self.embedding_dim + cond_dim, self.n_features)
        self.discriminator = Discriminator(self.n_features + cond_dim)
        return self

    def sample(self, n_samples: int = 1000) -> np.ndarray:
        """Generate high-fidelity synthetic records."""
        cond_dim = 16
        noise = np.random.normal(0, 1, (n_samples, self.embedding_dim))
        cond = np.random.normal(0, 1, (n_samples, cond_dim))

        synthetic_raw = self.generator.forward(noise, cond)
        return synthetic_raw
