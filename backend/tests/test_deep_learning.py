"""
ModelForge AI - PyTorch Tabular Deep Learning Unit Tests
"""

import numpy as np
import pytest
from ml_engine.algorithms.deep_learning import TabularMLP, PyTorchTabularModel, HAS_TORCH


def test_tabular_mlp_forward():
    if not HAS_TORCH:
        pytest.skip("PyTorch runtime not installed in environment")
    import torch
    model = TabularMLP(
        input_dim=10,
        output_dim=2,
        hidden_dims=[32, 16],
        dropout_rate=0.1,
        is_classification=True,
    )
    x = torch.randn(16, 10)
    out = model(x)
    assert out.shape == (16, 2)


def test_pytorch_tabular_training_classification():
    np.random.seed(42)
    X = np.random.randn(100, 8)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = PyTorchTabularModel(
        hidden_dims=[16, 8],
        learning_rate=0.01,
        batch_size=16,
        epochs=5,
        is_classification=True,
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    assert len(preds) == 100
    assert probs.shape == (100, 2)


def test_pytorch_tabular_training_regression():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] * 2.0 + X[:, 1] * 1.5

    reg = PyTorchTabularModel(
        hidden_dims=[16, 8],
        learning_rate=0.01,
        batch_size=16,
        epochs=5,
        is_classification=False,
    )
    reg.fit(X, y)
    preds = reg.predict(X)
    assert len(preds) == 100
    assert preds.ndim == 1
