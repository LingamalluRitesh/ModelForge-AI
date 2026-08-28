"""
ModelForge AI - Anomaly Detection Unit Tests
"""

import numpy as np
import pytest
from ml_engine.algorithms.anomaly_detection import (
    IsolationForestAnomalyDetector,
    LocalOutlierFactorDetector,
    OneClassSVMDetector,
)


def test_isolation_forest_anomaly_detection():
    np.random.seed(42)
    inliers = np.random.normal(0, 1, size=(200, 4))
    outliers = np.random.uniform(10, 15, size=(10, 4))
    X = np.vstack([inliers, outliers])

    detector = IsolationForestAnomalyDetector(contamination=0.05)
    detector.fit(X)
    preds = detector.predict(X)
    scores = detector.score_samples(X)

    assert len(preds) == 210
    # Outliers should have high scores
    assert np.mean(scores[200:]) > np.mean(scores[:200])


def test_local_outlier_factor():
    np.random.seed(42)
    inliers = np.random.normal(0, 1, size=(100, 3))
    outliers = np.random.uniform(8, 12, size=(5, 3))
    X = np.vstack([inliers, outliers])

    detector = LocalOutlierFactorDetector(contamination=0.05)
    detector.fit(X)
    preds = detector.predict(X)

    assert len(preds) == 105
    assert np.sum(preds) > 0


def test_one_class_svm():
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(100, 3))
    detector = OneClassSVMDetector(nu=0.05)
    detector.fit(X)
    preds = detector.predict(X)

    assert len(preds) == 100
