"""
ModelForge AI - Drift Detection Unit Tests
"""

import numpy as np
import pandas as pd
import pytest
from ml_engine.drift.drift_detector import DriftDetector


def test_psi_identical_distributions():
    np.random.seed(42)
    base = np.random.normal(0, 1, 1000)
    target = np.random.normal(0, 1, 1000)

    psi = DriftDetector.calculate_psi(base, target)
    assert psi < 0.10  # No significant drift


def test_psi_drifted_distributions():
    np.random.seed(42)
    base = np.random.normal(0, 1, 1000)
    target = np.random.normal(2.5, 1, 1000)  # Significant shift in mean

    psi = DriftDetector.calculate_psi(base, target)
    assert psi >= 0.25  # Significant drift detected


def test_feature_drift_evaluation():
    np.random.seed(42)
    baseline_df = pd.DataFrame({
        "feature_stable": np.random.normal(10, 2, 500),
        "feature_drifted": np.random.normal(50, 5, 500),
    })

    target_df = pd.DataFrame({
        "feature_stable": np.random.normal(10, 2, 500),
        "feature_drifted": np.random.normal(85, 15, 500),  # Drifted
    })

    results = DriftDetector.evaluate_feature_drift(baseline_df, target_df)
    assert results["drifted_features_count"] >= 1
    assert results["feature_metrics"]["feature_drifted"]["has_drifted"] is True
    assert results["feature_metrics"]["feature_stable"]["has_drifted"] is False
