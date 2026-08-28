"""
ModelForge AI - SHAP Explainability Unit Tests
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from ml_engine.explainability.shap_engine import ExplainabilityEngine


def test_shap_global_importances():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = ((X[:, 0] * 3.0 + X[:, 1]) > 0).astype(int)
    feature_names = ["dominant_feat", "secondary_feat", "noise_1", "noise_2"]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    result = ExplainabilityEngine.compute_global_explanations(model=model, X_sample=X, feature_names=feature_names)
    importances = result["feature_importances"]

    assert "dominant_feat" in importances
    assert "secondary_feat" in importances
    assert importances["dominant_feat"] > importances["noise_1"]


def test_shap_local_prediction():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = ((X[:, 0] * 3.0 + X[:, 1]) > 0).astype(int)
    feature_names = ["feat_1", "feat_2", "feat_3", "feat_4"]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    sample = {name: float(val) for name, val in zip(feature_names, X[0])}
    explanation = ExplainabilityEngine.explain_local_instance(model=model, features=sample, feature_names=feature_names)

    assert "prediction_value" in explanation
    assert "factors" in explanation
    assert len(explanation["factors"]) == 4
