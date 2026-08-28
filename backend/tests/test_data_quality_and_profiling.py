"""
ModelForge AI - Data Quality & Profiling Unit Tests
"""

import numpy as np
import pandas as pd
import pytest
from ml_engine.data_quality.quality_engine import DataQualityEngine
from ml_engine.profiling.statistical_profiler import StatisticalProfiler


@pytest.fixture
def sample_clean_df():
    np.random.seed(42)
    df = pd.DataFrame({
        "customer_id": [f"c_{i}" for i in range(100)],
        "age": np.random.randint(18, 70, size=100),
        "income": np.random.uniform(20000, 150000, size=100),
        "credit_score": np.random.uniform(300, 850, size=100),
        "is_active": np.random.choice([0, 1], size=100),
    })
    return df


@pytest.fixture
def sample_dirty_df():
    np.random.seed(42)
    df = pd.DataFrame({
        "customer_id": [f"c_{i//2}" for i in range(100)], # lots of duplicate customer IDs
        "age": [np.nan if i % 10 == 0 else float(np.random.randint(18, 70)) for i in range(100)], # missing
        "income": [1e8 if i == 0 else np.random.uniform(20000, 150000) for i in range(100)], # extreme outlier
        "constant_col": [1 for _ in range(100)], # constant
    })
    return df


def test_data_quality_clean(sample_clean_df):
    engine = DataQualityEngine()
    results = engine.evaluate_quality(sample_clean_df)
    assert results["quality_score"] >= 90.0
    assert results["passed_rules"] > 0
    assert results["failed_rules"] == 0
    assert results["missing_value_percentage"] == 0.0


def test_data_quality_dirty(sample_dirty_df):
    engine = DataQualityEngine()
    results = engine.evaluate_quality(sample_dirty_df)
    assert results["quality_score"] <= 95.0
    assert "constant_columns" in results["anomalies"]


def test_statistical_profiler(sample_clean_df):
    profile = StatisticalProfiler.profile_dataset(sample_clean_df)
    assert "column_stats" in profile
    assert "age" in profile["column_stats"]
    assert "income" in profile["column_stats"]
    assert "mean" in profile["column_stats"]["age"]
    assert "quantiles" in profile["column_stats"]["income"]
    assert "correlations" in profile
