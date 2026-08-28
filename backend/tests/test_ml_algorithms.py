"""
ModelForge AI - ML Engine Unit Tests
Verifies Classification, Regression, Clustering, PyTorch Deep Learning, and Evaluation Metrics.
"""

import numpy as np
import pandas as pd
import pytest
from ml_engine.algorithms.classification import (
    LogisticRegressionClassifier, DecisionTreeModel, RandomForestModel,
    GradientBoostingModel, XGBoostClassifierModel, LightGBMClassifierModel
)
from ml_engine.algorithms.regression import (
    LinearRegressionModel, RandomForestRegressorModel, XGBoostRegressorModel
)
from ml_engine.algorithms.clustering import (
    KMeansClusteringModel, DBSCANClusteringModel
)
from ml_engine.algorithms.deep_learning import PyTorchTabularModel
from ml_engine.evaluation.classification_evaluator import (
    ClassificationEvaluator, RegressionEvaluator
)


@pytest.fixture
def binary_classification_data():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    # Simple linear decision boundary with noise
    y = ((X[:, 0] * 2 + X[:, 1] - X[:, 2]) > 0).astype(int)
    return X, y


@pytest.fixture
def regression_data():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = X[:, 0] * 3.5 + X[:, 1] * 2.0 - X[:, 2] * 1.5 + np.random.randn(200) * 0.1
    return X, y


def test_logistic_regression(binary_classification_data):
    X, y = binary_classification_data
    clf = LogisticRegressionClassifier(C=1.0)
    clf.fit(X, y)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    assert len(preds) == len(y)
    assert probs.shape == (200, 2)
    metrics = ClassificationEvaluator.evaluate(y, preds, probs)
    assert metrics["accuracy"] > 0.80
    assert metrics["f1"] > 0.80
    assert metrics["roc_auc"] > 0.85


def test_random_forest_classifier(binary_classification_data):
    X, y = binary_classification_data
    clf = RandomForestModel(n_estimators=30, max_depth=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    metrics = ClassificationEvaluator.evaluate(y, preds)
    assert metrics["accuracy"] > 0.85


def test_xgboost_classifier(binary_classification_data):
    X, y = binary_classification_data
    clf = XGBoostClassifierModel(n_estimators=20, max_depth=4)
    clf.fit(X, y)
    preds = clf.predict(X)
    metrics = ClassificationEvaluator.evaluate(y, preds)
    assert metrics["accuracy"] > 0.85


def test_lightgbm_classifier(binary_classification_data):
    X, y = binary_classification_data
    clf = LightGBMClassifierModel(n_estimators=20, max_depth=4)
    clf.fit(X, y)
    preds = clf.predict(X)
    metrics = ClassificationEvaluator.evaluate(y, preds)
    assert metrics["accuracy"] > 0.85


def test_pytorch_tabular_model(binary_classification_data):
    X, y = binary_classification_data
    model = PyTorchTabularModel(hidden_dims=[32, 16], epochs=10, is_classification=True)
    model.fit(X, y)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    assert len(preds) == len(y)
    assert probs.shape == (200, 2)


def test_regression_models(regression_data):
    X, y = regression_data
    reg = LinearRegressionModel()
    reg.fit(X, y)
    preds = reg.predict(X)
    metrics = RegressionEvaluator.evaluate(y, preds)
    assert metrics["r2"] > 0.95
    assert metrics["rmse"] < 0.5


def test_kmeans_clustering():
    np.random.seed(42)
    X = np.random.randn(150, 4)
    kmeans = KMeansClusteringModel(n_clusters=3)
    labels = kmeans.fit_predict(X)
    assert len(labels) == 150
    metrics = kmeans.compute_clustering_metrics(X)
    assert metrics["n_clusters"] == 3
    assert "silhouette_score" in metrics
