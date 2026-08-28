"""
ModelForge AI - ML Engine: Graph Neural Networks for Fraud Detection (GNN / GraphSAGE)
Implements Graph Convolutional Networks (GCN) and GraphSAGE neighborhood aggregation
to detect syndicate fraud, money laundering clusters, and multi-account identity theft.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from app.core.exceptions import MLModelExecutionException
from app.core.logging import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object


if HAS_TORCH:
    class GraphSageLayer(nn.Module):
        """GraphSAGE layer with mean aggregator and non-linear feature projection."""

        def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.dropout = dropout

            self.weight_self = nn.Parameter(torch.Tensor(in_features, out_features))
            self.weight_neigh = nn.Parameter(torch.Tensor(in_features, out_features))
            self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight_self)
            nn.init.xavier_uniform_(self.weight_neigh)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
            # x: [N, in_features], adj_norm: [N, N] normalized adjacency matrix
            h_self = torch.mm(x, self.weight_self)
            neigh_agg = torch.spmm(adj_norm, x) if adj_norm.is_sparse else torch.mm(adj_norm, x)
            h_neigh = torch.mm(neigh_agg, self.weight_neigh)

            h = h_self + h_neigh
            if self.bias is not None:
                h = h + self.bias

            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # L2 normalize node representations
            h = F.normalize(h, p=2, dim=1)
            return h


    class FraudGNNArchitecture(nn.Module):
        """Multi-layer GraphSAGE classifier for transaction-node fraud classification."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            output_dim: int = 2,
            n_layers: int = 2,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(GraphSageLayer(input_dim, hidden_dim, dropout=dropout))

            for _ in range(n_layers - 2):
                self.layers.append(GraphSageLayer(hidden_dim, hidden_dim, dropout=dropout))

            self.classifier = nn.Linear(hidden_dim, output_dim)

        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            h = x
            for layer in self.layers:
                h = layer(h, adj)
            logits = self.classifier(h)
            return logits


class FraudGNNModel:
    """Production Graph Neural Network runner for transaction and entity fraud analysis."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        learning_rate: float = 0.005,
        weight_decay: float = 5e-4,
        epochs: int = 40,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_state = random_state

        self.model = None
        self.device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        self.is_fitted = False
        self._linear_weights = None

    @staticmethod
    def build_adjacency_matrix(
        edge_index: np.ndarray,
        num_nodes: int,
        add_self_loops: bool = True,
    ) -> np.ndarray:
        """Construct normalized symmetric Laplacian adjacency matrix $D^{-1/2} (A + I) D^{-1/2}$."""
        adj = sp.coo_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes),
            dtype=np.float32,
        )
        # Symmetrize
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        if add_self_loops:
            adj = adj + sp.eye(adj.shape[0])

        row_sum = np.array(adj.sum(1))
        deg_inv_sqrt = np.power(row_sum, -0.5).flatten()
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        deg_mat_inv_sqrt = sp.diags(deg_inv_sqrt)

        adj_normalized = deg_mat_inv_sqrt.dot(adj).dot(deg_mat_inv_sqrt).tocoo()
        return adj_normalized.toarray()

    def fit(
        self,
        node_features: np.ndarray,
        labels: np.ndarray,
        edge_index: Optional[np.ndarray] = None,
        train_mask: Optional[np.ndarray] = None,
    ):
        num_nodes = node_features.shape[0]
        input_dim = node_features.shape[1]

        if edge_index is None:
            # Construct k-nearest neighbor transaction graph if edges not explicitly provided
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(6, num_nodes), algorithm="auto").fit(node_features)
            _, indices = nbrs.kneighbors(node_features)
            rows = np.repeat(np.arange(num_nodes), indices.shape[1])
            cols = indices.flatten()
            edge_index = np.vstack([rows, cols])

        adj_matrix = self.build_adjacency_matrix(edge_index, num_nodes)

        if not HAS_TORCH:
            X_bias = np.column_stack([np.ones(num_nodes), node_features])
            self._linear_weights = np.linalg.pinv(X_bias.T @ X_bias + 1e-3 * np.eye(X_bias.shape[1])) @ X_bias.T @ labels
            self.is_fitted = True
            return self

        torch.manual_seed(self.random_state)
        self.model = FraudGNNArchitecture(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=len(np.unique(labels)),
            n_layers=self.n_layers,
        ).to(self.device)

        x_t = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        adj_t = torch.tensor(adj_matrix, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(labels, dtype=torch.long).to(self.device)

        if train_mask is None:
            mask_t = torch.ones(num_nodes, dtype=torch.bool, device=self.device)
        else:
            mask_t = torch.tensor(train_mask, dtype=torch.bool, device=self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(x_t, adj_t)
            loss = criterion(logits[mask_t], y_t[mask_t])
            loss.backward()
            optimizer.step()

        self.is_fitted = True
        return self

    def predict(self, node_features: np.ndarray, adj_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise MLModelExecutionException("FraudGNNModel is not fitted.")

        if not HAS_TORCH or self.model is None:
            X_bias = np.column_stack([np.ones(node_features.shape[0]), node_features])
            raw = X_bias @ self._linear_weights
            return (raw >= 0.5).astype(int)

        if adj_matrix is None:
            adj_matrix = np.eye(node_features.shape[0], dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(node_features, dtype=torch.float32).to(self.device)
            adj_t = torch.tensor(adj_matrix, dtype=torch.float32).to(self.device)
            logits = self.model(x_t, adj_t)
            return torch.argmax(logits, dim=1).cpu().numpy()
