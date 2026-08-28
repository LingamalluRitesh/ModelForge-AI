"""
ModelForge AI - Algorithmic Fairness & Bias Audit Unit Tests
"""

import numpy as np
import pytest
from ml_engine.fairness.fairness_engine import FairnessAuditEngine


def test_fairness_audit_demographic_parity():
    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1], size=n)
    y_pred = np.random.choice([0, 1], size=n)
    sensitive_attr = np.random.choice([0, 1], size=n)

    report = FairnessAuditEngine.audit_model_fairness(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_attribute_values=sensitive_attr,
    )

    assert "demographic_parity_ratio" in report
    assert "disparate_impact_ratio" in report
    assert "equal_opportunity_difference" in report
    assert "is_fair" in report
    assert isinstance(report["is_fair"], (bool, np.bool_))
