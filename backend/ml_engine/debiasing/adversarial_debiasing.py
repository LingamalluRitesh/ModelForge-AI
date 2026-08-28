"""
ModelForge AI - ML Engine: Adversarial Debiasing
In-processing neural network debiasing with Gradient Reversal Layer (GRL).
Learns a latent representation $Z$ that maximizes task prediction accuracy while
minimizing the adversary's capability to reconstruct sensitive protected attributes.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from app.core.exceptions import MLModelExecutionException
from app.core.logging import logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object


if HAS_TORCH:
    class GradReverse(torch.autograd.Function):
        """Gradient Reversal Layer (GRL). Passes forward unchanged, negates gradients during backward."""

        @staticmethod
        def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
            ctx.alpha = alpha
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
            return grad_output.neg() * ctx.alpha, None


    class AdversarialDebiasingArchitecture(nn.Module):
        """Dual-head neural network: Predictor Network and Adversary Discriminator."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int = 2,
            hidden_dim: int = 64,
            adversary_weight: float = 1.0,
        ):
            super().__init__()
            self.adversary_weight = adversary_weight

            # Feature Extractor
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )

            # Target Task Classifier
            self.predictor = nn.Linear(hidden_dim // 2, output_dim)

            # Protected Attribute Adversary
            self.adversary = nn.Sequential(
                nn.Linear(hidden_dim // 2, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def forward(self, x: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
            latent = self.feature_extractor(x)
            pred_logits = self.predictor(latent)

            # Reverse gradient flowing into adversary
            rev_latent = GradReverse.apply(latent, alpha)
            adv_logits = self.adversary(rev_latent)

            return pred_logits, adv_logits


class AdversarialDebiasingClassifier:
    """Production estimator wrapper for In-Processing Adversarial Debiasing."""

    def __init__(
        self,
        hidden_dim: int = 64,
        adversary_weight: float = 1.0,
        learning_rate: float = 0.001,
        epochs: int = 25,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.adversary_weight = adversary_weight
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.model = None
        self.device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        self.is_fitted = False
        self._linear_weights = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sensitive_attribute: Union[np.ndarray, pd.Series],
    ):
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        y_arr = np.asarray(y)
        s_arr = np.asarray(sensitive_attribute)

        if not HAS_TORCH:
            X_bias = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
            self._linear_weights = np.linalg.pinv(X_bias.T @ X_bias + 1e-3 * np.eye(X_bias.shape[1])) @ X_bias.T @ y_arr
            self.is_fitted = True
            return self

        torch.manual_seed(self.random_state)
        input_dim = X_arr.shape[1]
        output_dim = len(np.unique(y_arr))

        self.model = AdversarialDebiasingArchitecture(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=self.hidden_dim,
            adversary_weight=self.adversary_weight,
        ).to(self.device)

        X_t = torch.tensor(X_arr, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.long)
        s_t = torch.tensor(s_arr, dtype=torch.long)

        dataset = TensorDataset(X_t, y_t, s_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        pred_loss_fn = nn.CrossEntropyLoss()
        adv_loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            # Dynamic adversarial schedule parameter
            p = float(epoch) / float(self.epochs)
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

            for bx, by, bs in loader:
                bx, by, bs = bx.to(self.device), by.to(self.device), bs.to(self.device)
                optimizer.zero_grad()
                pred_out, adv_out = self.model(bx, alpha)
                loss = pred_loss_fn(pred_out, by) + self.adversary_weight * adv_loss_fn(adv_out, bs)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise MLModelExecutionException("AdversarialDebiasingClassifier is not fitted.")

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
            X_t = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
            pred_out, _ = self.model(X_t)
            return torch.argmax(pred_out, dim=1).cpu().numpy()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise MLModelExecutionException("AdversarialDebiasingClassifier is not fitted.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if not HAS_TORCH or self.model is None:
            preds = self.predict(X_arr)
            return np.column_stack([1.0 - preds, preds])

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
            pred_out, _ = self.model(X_t)
            return torch.softmax(pred_out, dim=1).cpu().numpy()
