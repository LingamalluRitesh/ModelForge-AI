"""
ModelForge AI - ML Engine: GraphSAGE Inductive Node Representation Learning
Implements Hamilton, Ying, & Leskovec Inductive Representation Learning on Large Graphs (GraphSAGE)
with Uniform Neighborhood Sampling, Mean / LSTM / Pooling Aggregators, and Multi-Hop Embeddings.
$h_{\mathcal{N}(v)}^{(k)} = 	ext{AGGREGATE}_k \left(\{h_u^{(k-1)}, orall u \in \mathcal{N}(v)\}ight)$
$h_v^{(k)} = \sigma\left(W^{(k)} \cdot \left[ h_v^{(k-1)} || h_{\mathcal{N}(v)}^{(k)} ight]ight)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class MeanAggregator:
    """Computes element-wise mean over sample neighbor embeddings."""
    def aggregate(self, neighbor_embeddings: np.ndarray) -> np.ndarray:
        return np.mean(neighbor_embeddings, axis=0)


class MaxPoolAggregator:
    """Multi-layer perceptron feature transformation followed by element-wise max-pooling."""
    def __init__(self, in_features: int, hidden_dim: int):
        self.W_pool = np.random.normal(0, np.sqrt(2.0 / in_features), (in_features, hidden_dim))
        self.b_pool = np.zeros(hidden_dim)

    def aggregate(self, neighbor_embeddings: np.ndarray) -> np.ndarray:
        transformed = np.maximum(0, np.dot(neighbor_embeddings, self.W_pool) + self.b_pool)
        return np.max(transformed, axis=0)


class GraphSAGELayer:
    """Single GraphSAGE convolution layer with concatenation and L2 normalization."""
    def __init__(self, in_features: int, out_features: int, aggregator_type: str = "mean"):
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator_type = aggregator_type

        if aggregator_type == "pool":
            self.aggregator = MaxPoolAggregator(in_features, in_features)
        else:
            self.aggregator = MeanAggregator()

        # Weight matrix for concatenated [self || neighbor_agg]
        std = np.sqrt(2.0 / (2 * in_features))
        self.W = np.random.normal(0, std, (2 * in_features, out_features))

    def forward(self, self_embedding: np.ndarray, neighbor_embeddings: np.ndarray) -> np.ndarray:
        agg = self.aggregator.aggregate(neighbor_embeddings)
        concat = np.concatenate([self_embedding, agg])
        h_new = np.maximum(0, np.dot(concat, self.W))
        norm = np.linalg.norm(h_new)
        return h_new / max(1e-8, norm)


class GraphSAGEModel:
    """Multi-layer GraphSAGE Inductive Embedding Engine."""
    def __init__(self, in_features: int, hidden_dims: List[int], num_samples_per_hop: List[int] = [10, 5]):
        self.num_samples_per_hop = num_samples_per_hop
        self.layers = []
        curr_dim = in_features

        for h_dim in hidden_dims:
            self.layers.append(GraphSAGELayer(curr_dim, h_dim))
            curr_dim = h_dim

    def sample_neighbors(self, adj_list: Dict[int, List[int]], node_id: int, num_samples: int) -> List[int]:
        neighbors = adj_list.get(node_id, [node_id])
        if len(neighbors) == 0:
            return [node_id]
        return list(np.random.choice(neighbors, size=num_samples, replace=True))
