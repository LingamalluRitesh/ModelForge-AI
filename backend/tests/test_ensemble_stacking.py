"""
ModelForge AI - Ensemble Stacking Classifier Unit Tests
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from ml_engine.algorithms.ensemble_stacking import StackingClassifierModel


def test_stacking_classifier():
    np.random.seed(42)
    X = np.random.randn(150, 4)
    y = ((X[:, 0] * 2 + X[:, 1] - X[:, 2]) > 0).astype(int)

    base_models = [
        RandomForestClassifier(n_estimators=10, random_state=42),
        LogisticRegression(C=1.0),
    ]
    meta_model = LogisticRegression(C=1.0)

    stacker = StackingClassifierModel(base_models=base_models, meta_model=meta_model, n_splits=3)
    stacker.fit(X, y)

    preds = stacker.predict(X)
    probs = stacker.predict_proba(X)

    assert len(preds) == 150
    assert probs.shape == (150, 2)
    acc = np.mean(preds == y)
    assert acc > 0.80
