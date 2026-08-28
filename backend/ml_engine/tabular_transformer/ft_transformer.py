"""
ModelForge AI - ML Engine: Feature Tokenizer Transformer (FT-Transformer)
Production-grade PyTorch & NumPy implementation of the Feature Tokenizer Transformer architecture
for high-performance tabular data modeling with continuous and categorical embeddings,
multi-head self-attention blocks, and residual feed-forward layers.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd

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

from app.core.exceptions import MLModelExecutionException
from app.core.logging import logger


if HAS_TORCH:
    class NumericalFeatureTokenizer(nn.Module):
        """Projects continuous numerical features into dense vector embeddings via affine transformations."""

        def __init__(self, num_numerical_features: int, d_token: int, bias: bool = True):
            super().__init__()
            self.num_features = num_numerical_features
            self.d_token = d_token
            self.weight = nn.Parameter(torch.Tensor(num_numerical_features, d_token))
            self.bias = nn.Parameter(torch.Tensor(num_numerical_features, d_token)) if bias else None
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: [batch_size, num_features]
            # output shape: [batch_size, num_features, d_token]
            x = x.unsqueeze(-1)
            tokens = x * self.weight.unsqueeze(0)
            if self.bias is not None:
                tokens = tokens + self.bias.unsqueeze(0)
            return tokens


    class CategoricalFeatureTokenizer(nn.Module):
        """Embeds categorical discrete features into uniform dense token representations."""

        def __init__(self, cardinalities: List[int], d_token: int):
            super().__init__()
            self.embeddings = nn.ModuleList([
                nn.Embedding(card, d_token) for card in cardinalities
            ])
            self.d_token = d_token

        def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
            # x_cat shape: [batch_size, num_categorical_features]
            tokens = [emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.embeddings)]
            return torch.cat(tokens, dim=1) if tokens else torch.empty(x_cat.shape[0], 0, self.d_token, device=x_cat.device)


    class TransformerBlock(nn.Module):
        """Standard Pre-LayerNorm Transformer Encoder block with Multi-Head Attention and FFN."""

        def __init__(
            self,
            d_token: int,
            n_heads: int = 8,
            d_ffn_factor: float = 4.0,
            attention_dropout: float = 0.1,
            ffn_dropout: float = 0.1,
            activation: str = "gelu",
        ):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_token)
            self.attention = nn.MultiheadAttention(
                embed_dim=d_token,
                num_heads=n_heads,
                dropout=attention_dropout,
                batch_first=True,
            )
            self.norm2 = nn.LayerNorm(d_token)
            d_ffn = int(d_token * d_ffn_factor)
            
            act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
            self.ffn = nn.Sequential(
                nn.Linear(d_token, d_ffn),
                act_fn,
                nn.Dropout(ffn_dropout),
                nn.Linear(d_ffn, d_token),
                nn.Dropout(ffn_dropout),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            norm_x = self.norm1(x)
            attn_out, attn_weights = self.attention(norm_x, norm_x, norm_x)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, attn_weights


    class FTTransformerArchitecture(nn.Module):
        """Full Feature Tokenizer Transformer architecture with [CLS] token and classification/regression head."""

        def __init__(
            self,
            num_numerical: int,
            categorical_cardinalities: List[int],
            output_dim: int,
            d_token: int = 192,
            n_blocks: int = 3,
            n_heads: int = 8,
            d_ffn_factor: float = 4.0,
            attention_dropout: float = 0.1,
            ffn_dropout: float = 0.1,
            residual_dropout: float = 0.0,
        ):
            super().__init__()
            self.num_numerical = num_numerical
            self.num_categorical = len(categorical_cardinalities)

            if num_numerical > 0:
                self.num_tokenizer = NumericalFeatureTokenizer(num_numerical, d_token)
            else:
                self.num_tokenizer = None

            if self.num_categorical > 0:
                self.cat_tokenizer = CategoricalFeatureTokenizer(categorical_cardinalities, d_token)
            else:
                self.cat_tokenizer = None

            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.normal_(self.cls_token, std=0.02)

            self.blocks = nn.ModuleList([
                TransformerBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    d_ffn_factor=d_ffn_factor,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                )
                for _ in range(n_blocks)
            ])

            self.head_norm = nn.LayerNorm(d_token)
            self.head = nn.Linear(d_token, output_dim)

        def forward(
            self,
            x_num: Optional[torch.Tensor] = None,
            x_cat: Optional[torch.Tensor] = None,
            return_attention: bool = False,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
            batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
            tokens_list = []

            # 1. Add [CLS] token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens_list.append(cls_tokens)

            # 2. Tokenize continuous numerical features
            if self.num_tokenizer is not None and x_num is not None:
                num_tokens = self.num_tokenizer(x_num)
                tokens_list.append(num_tokens)

            # 3. Tokenize categorical features
            if self.cat_tokenizer is not None and x_cat is not None:
                cat_tokens = self.cat_tokenizer(x_cat)
                tokens_list.append(cat_tokens)

            # Concatenate all tokens along sequence dimension
            x = torch.cat(tokens_list, dim=1)

            attention_maps = []
            for block in self.blocks:
                x, attn_weights = block(x)
                if return_attention:
                    attention_maps.append(attn_weights)

            # Extract final representation of [CLS] token
            cls_repr = x[:, 0]
            cls_repr = self.head_norm(cls_repr)
            logits = self.head(cls_repr)

            if return_attention:
                return logits, attention_maps
            return logits


class FTTransformerModel:
    """Production estimator wrapper for FT-Transformer with scikit-learn compatible fit/predict APIs."""

    def __init__(
        self,
        d_token: int = 128,
        n_blocks: int = 3,
        n_heads: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 128,
        epochs: int = 25,
        problem_type: str = "classification",
        random_state: int = 42,
    ):
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.problem_type = problem_type
        self.random_state = random_state

        self.model = None
        self.device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        self.is_fitted = False
        self.feature_names_: List[str] = []
        self._linear_weights = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        categorical_indices: Optional[List[int]] = None,
    ):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            self.feature_names_ = [f"feat_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y)

        if not HAS_TORCH:
            # Analytical Ridge solver fallback when PyTorch is not present
            X_bias = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
            self._linear_weights = np.linalg.pinv(X_bias.T @ X_bias + 1e-3 * np.eye(X_bias.shape[1])) @ X_bias.T @ y_arr
            self.is_fitted = True
            return self

        torch.manual_seed(self.random_state)
        num_numerical = X_arr.shape[1]
        output_dim = len(np.unique(y_arr)) if self.problem_type == "classification" else 1

        self.model = FTTransformerArchitecture(
            num_numerical=num_numerical,
            categorical_cardinalities=[],
            output_dim=output_dim,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
        ).to(self.device)

        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        if self.problem_type == "classification":
            y_tensor = torch.tensor(y_arr, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
        else:
            y_tensor = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
            criterion = nn.MSELoss()

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x_num=batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise MLModelExecutionException("FT-Transformer model is not fitted.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if not HAS_TORCH or self.model is None:
            X_bias = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
            raw = X_bias @ self._linear_weights
            if self.problem_type == "classification":
                return (raw >= 0.5).astype(int)
            return raw

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
            logits = self.model(x_num=X_tensor)
            if self.problem_type == "classification":
                return torch.argmax(logits, dim=1).cpu().numpy()
            return logits.squeeze(1).cpu().numpy()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.problem_type != "classification":
            raise MLModelExecutionException("predict_proba is only supported for classification problems.")

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
            logits = self.model(x_num=X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs
