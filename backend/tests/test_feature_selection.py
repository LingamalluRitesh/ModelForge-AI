"""
ModelForge AI - Feature Selection & Dimensionality Reduction Unit Tests
"""

import numpy as np
import pytest
from ml_engine.preprocessing.feature_selection import AdvancedFeatureSelector


def test_feature_selector_variance_and_mutual_info():
    np.random.seed(42)
    # 2 informative features, 1 constant (0 variance), 1 highly correlated duplicate
    informative_1 = np.random.normal(0, 1, 100)
    informative_2 = np.random.normal(0, 1, 100)
    constant_col = np.zeros(100)
    duplicate_col = informative_1 * 0.999 + np.random.normal(0, 0.0001, 100)
    y = ((informative_1 + informative_2) > 0).astype(int)

    X = np.column_stack([informative_1, informative_2, constant_col, duplicate_col])
    names = ["info_1", "info_2", "constant", "duplicate"]

    selector = AdvancedFeatureSelector(variance_threshold=0.01, max_correlation=0.95, top_k_features=2)
    X_selected = selector.fit_transform(X, y, names)

    assert X_selected.shape[1] == 2
    assert "constant" not in selector.selected_feature_names_
    assert "duplicate" not in selector.selected_feature_names_
