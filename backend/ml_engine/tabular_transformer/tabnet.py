"""
ModelForge AI - ML Engine: TabNet Architecture
Production-grade TabNet (Attentive Interpretable Tabular Learning) implementing
Sparsemax feature selection masks, sequential multi-step decision blocks,
prior scales for feature reuse control, and native interpretability attribution masks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from app.core.exceptions import MLModelExecutionException
from app.core.logging import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object


if HAS_TORCH:
    class SparsemaxFunction(torch.autograd.Function):
        """Sparsemax activation function computing sparse Euclidean projections onto the simplex."""

        @staticmethod
        def forward(ctx, input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
            ctx.dim = dim
            input_dim = input_tensor.dim()
            dim = dim if dim >= 0 else input_dim + dim

            sorted_tensor, _ = torch.sort(input_tensor, descending=True, dim=dim)
            range_values = torch.arange(1, input_tensor.size(dim) + 1, device=input_tensor.device, dtype=input_tensor.dtype)
            view_shape = [1] * input_dim
            view_shape[dim] = -1
            range_values = range_values.view(*view_shape)

            cumsum_sorted = torch.cumsum(sorted_tensor, dim=dim)
            condition = 1 + range_values * sorted_tensor > cumsum_sorted
            k_indices = torch.max(range_values * condition.float(), dim=dim, keepdim=True)[0]
            tau_z = (torch.gather(cumsum_sorted, dim, (k_indices - 1).long()) - 1) / k_indices
            output = torch.clamp(input_tensor - tau_z, min=0)
            ctx.save_for_backward(output)
            return output

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
            output, = ctx.saved_tensors
            non_zeros = (output > 0).float()
            sum_grad = torch.sum(grad_output * non_zeros, dim=ctx.dim, keepdim=True)
            non_zero_count = torch.sum(non_zeros, dim=ctx.dim, keepdim=True) + 1e-10
            v_hat = sum_grad / non_zero_count
            grad_input = non_zeros * (grad_output - v_hat)
            return grad_input, None


    class Sparsemax(nn.Module):
        def __init__(self, dim: int = -1):
            super().__init__()
            self.dim = dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return SparsemaxFunction.apply(x, self.dim)


    class GBN(nn.Module):
        """Ghost Batch Normalization for large-batch regularized tabular training."""

        def __init__(self, input_dim: int, virtual_batch_size: int = 128, momentum: float = 0.02):
            super().__init__()
            self.input_dim = input_dim
            self.virtual_batch_size = virtual_batch_size
            self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] <= self.virtual_batch_size:
                return self.bn(x)
            chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), dim=0)
            res = [self.bn(c) for c in chunks]
            return torch.cat(res, dim=0)


    class FeatureTransformerBlock(nn.Module):
        """Shared and decision-step dependent Feature Transformer layers with GLU activations."""

        def __init__(self, input_dim: int, output_dim: int, virtual_batch_size: int = 128, momentum: float = 0.02):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim * 2, bias=False)
            self.gbn = GBN(output_dim * 2, virtual_batch_size=virtual_batch_size, momentum=momentum)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear(x)
            x = self.gbn(x)
            # Gated Linear Unit (GLU)
            return F.glu(x, dim=-1) * math.sqrt(0.5)


    class AttentiveTransformer(nn.Module):
        """Generates sparse feature selection masks $M[i]$ for decision step $i$."""

        def __init__(self, input_dim: int, output_dim: int, virtual_batch_size: int = 128, momentum: float = 0.02):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
            self.gbn = GBN(output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)
            self.sparsemax = Sparsemax(dim=-1)

        def forward(self, priors: torch.Tensor, processed_feat: torch.Tensor) -> torch.Tensor:
            x = self.linear(processed_feat)
            x = self.gbn(x)
            x = x * priors
            mask = self.sparsemax(x)
            return mask


    class TabNetArchitecture(nn.Module):
        """TabNet core neural architecture."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_d: int = 64,
            n_a: int = 64,
            n_steps: int = 5,
            gamma: float = 1.3,
            n_independent: int = 2,
            n_shared: int = 2,
            virtual_batch_size: int = 128,
            momentum: float = 0.02,
            mask_type: str = "sparsemax",
        ):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.n_d = n_d
            self.n_a = n_a
            self.n_steps = n_steps
            self.gamma = gamma
            self.virtual_batch_size = virtual_batch_size

            self.initial_bn = nn.BatchNorm1d(input_dim, momentum=0.01)

            # Shared Feature Transformer blocks
            self.shared_blocks = nn.ModuleList([
                FeatureTransformerBlock(
                    input_dim if i == 0 else (n_d + n_a),
                    n_d + n_a,
                    virtual_batch_size=virtual_batch_size,
                    momentum=momentum,
                )
                for i in range(n_shared)
            ])

            # Step-specific Feature Transformer and Attentive Transformer blocks
            self.step_blocks = nn.ModuleList()
            self.attentive_transformers = nn.ModuleList()

            for _ in range(n_steps):
                step_ft = nn.ModuleList([
                    FeatureTransformerBlock(
                        n_d + n_a,
                        n_d + n_a,
                        virtual_batch_size=virtual_batch_size,
                        momentum=momentum,
                    )
                    for _ in range(n_independent)
                ])
                self.step_blocks.append(step_ft)
                self.attentive_transformers.append(
                    AttentiveTransformer(
                        n_a,
                        input_dim,
                        virtual_batch_size=virtual_batch_size,
                        momentum=momentum,
                    )
                )

            self.final_mapping = nn.Linear(n_d, output_dim, bias=False)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
            res = self.initial_bn(x)
            batch_size = x.shape[0]

            prior = torch.ones((batch_size, self.input_dim), device=x.device)
            M_loss = 0.0
            att_masks = []
            out_d = torch.zeros((batch_size, self.n_d), device=x.device)

            # Initial representation
            x_feat = res
            for shared_b in self.shared_blocks:
                x_feat = shared_b(x_feat)

            att_a = x_feat[:, self.n_d:]

            for step in range(self.n_steps):
                M = self.attentive_transformers[step](prior, att_a)
                # Entropy regularization for sparsity
                M_loss += torch.mean(torch.sum(-M * torch.log(M + 1e-15), dim=-1))
                # Update feature reuse prior
                prior = prior * (self.gamma - M)
                att_masks.append(M)

                # Mask input features
                masked_x = M * res
                cur_feat = masked_x

                for shared_b in self.shared_blocks:
                    cur_feat = shared_b(cur_feat)

                for step_b in self.step_blocks[step]:
                    cur_feat = cur_feat + step_b(cur_feat)

                out_d = out_d + F.relu(cur_feat[:, :self.n_d])
                att_a = cur_feat[:, self.n_d:]

            logits = self.final_mapping(out_d)
            M_loss = M_loss / self.n_steps
            return logits, M_loss, att_masks


class TabNetClassifierModel:
    """Production TabNet Classifier with scikit-learn compatible interface."""

    def __init__(
        self,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 4,
        gamma: float = 1.3,
        lambda_sparse: float = 1e-3,
        learning_rate: float = 0.02,
        batch_size: int = 128,
        epochs: int = 20,
        random_state: int = 42,
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

        self.model = None
        self.device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        self.is_fitted = False
        self.feature_importances_: Optional[np.ndarray] = None
        self._linear_weights = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        y_arr = np.asarray(y)

        if not HAS_TORCH:
            X_bias = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
            self._linear_weights = np.linalg.pinv(X_bias.T @ X_bias + 1e-3 * np.eye(X_bias.shape[1])) @ X_bias.T @ y_arr
            self.feature_importances_ = np.ones(X_arr.shape[1]) / X_arr.shape[1]
            self.is_fitted = True
            return self

        torch.manual_seed(self.random_state)
        input_dim = X_arr.shape[1]
        output_dim = len(np.unique(y_arr))

        self.model = TabNetArchitecture(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
        ).to(self.device)

        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        y_tensor = torch.tensor(y_arr, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits, m_loss, _ = self.model(batch_x)
                loss = criterion(logits, batch_y) - self.lambda_sparse * m_loss
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        # Compute global feature importance via aggregated attention masks
        self.model.eval()
        with torch.no_grad():
            sample_tensor = torch.tensor(X_arr[:min(500, len(X_arr))], dtype=torch.float32).to(self.device)
            _, _, masks = self.model(sample_tensor)
            stacked_masks = torch.stack(masks, dim=0).mean(dim=[0, 1]).cpu().numpy()
            total = np.sum(stacked_masks) + 1e-12
            self.feature_importances_ = stacked_masks / total

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise MLModelExecutionException("TabNet model is not fitted.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if not HAS_TORCH or self.model is None:
            X_bias = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
            raw = X_bias @ self._linear_weights
            return (raw >= 0.5).astype(int)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
            logits, _, _ = self.model(X_tensor)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise MLModelExecutionException("TabNet model is not fitted.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if not HAS_TORCH or self.model is None:
            preds = self.predict(X_arr)
            return np.column_stack([1.0 - preds, preds])

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
            logits, _, _ = self.model(X_tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()
