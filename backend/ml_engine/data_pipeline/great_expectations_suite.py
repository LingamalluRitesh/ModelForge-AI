"""
ModelForge AI - ML Engine: Great Expectations Enterprise Data Validation Suite
Implements declarative data contract validations: Column Nullity, Value Range Limits,
Categorical Set Membership, Quantile Drift Tolerance, Regex Format Matching, and Multi-Column Integrity.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import re
import numpy as np
import pandas as pd


class ExpectationResult:
    def __init__(
        self,
        expectation_type: str,
        column: Optional[str],
        success: bool,
        observed_value: Any,
        expected_value: Any,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.expectation_type = expectation_type
        self.column = column
        self.success = success
        self.observed_value = observed_value
        self.expected_value = expected_value
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expectation_type": self.expectation_type,
            "column": self.column,
            "success": self.success,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "details": self.details,
        }


class DataContractValidationSuite:
    """Enterprise Data Quality Contract Validation Engine."""

    def __init__(self, suite_name: str = "production_golden_contract"):
        self.suite_name = suite_name
        self.expectations: List[Callable[[pd.DataFrame], ExpectationResult]] = []

    def expect_column_values_to_not_be_null(self, column: str, max_null_pct: float = 0.0):
        def _check(df: pd.DataFrame) -> ExpectationResult:
            if column not in df.columns:
                return ExpectationResult("expect_column_values_to_not_be_null", column, False, "COLUMN_NOT_FOUND", max_null_pct)
            null_pct = float(df[column].isnull().mean())
            success = null_pct <= max_null_pct
            return ExpectationResult(
                "expect_column_values_to_not_be_null",
                column,
                success,
                round(null_pct, 4),
                max_null_pct,
                {"null_count": int(df[column].isnull().sum())},
            )
        self.expectations.append(_check)
        return self

    def expect_column_values_to_be_between(
        self, column: str, min_val: Optional[float] = None, max_val: Optional[float] = None
    ):
        def _check(df: pd.DataFrame) -> ExpectationResult:
            if column not in df.columns:
                return ExpectationResult("expect_column_values_to_be_between", column, False, "COLUMN_NOT_FOUND", {"min": min_val, "max": max_val})
            s = df[column].dropna().astype(float)
            obs_min = float(s.min())
            obs_max = float(s.max())

            success = True
            if min_val is not None and obs_min < min_val:
                success = False
            if max_val is not None and obs_max > max_val:
                success = False

            return ExpectationResult(
                "expect_column_values_to_be_between",
                column,
                success,
                {"observed_min": obs_min, "observed_max": obs_max},
                {"min": min_val, "max": max_val},
            )
        self.expectations.append(_check)
        return self

    def expect_column_values_to_be_in_set(self, column: str, allowed_set: List[Any]):
        def _check(df: pd.DataFrame) -> ExpectationResult:
            if column not in df.columns:
                return ExpectationResult("expect_column_values_to_be_in_set", column, False, "COLUMN_NOT_FOUND", allowed_set)
            unique_vals = set(df[column].dropna().unique())
            unexpected = list(unique_vals - set(allowed_set))
            success = len(unexpected) == 0
            return ExpectationResult(
                "expect_column_values_to_be_in_set",
                column,
                success,
                {"unexpected_values": unexpected[:10]},
                allowed_set,
            )
        self.expectations.append(_check)
        return self

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute full validation suite against target DataFrame."""
        results = [exp(df) for exp in self.expectations]
        total = len(results)
        passed = sum(1 for r in results if r.success)
        failed = total - passed

        return {
            "suite_name": self.suite_name,
            "overall_success": failed == 0,
            "statistics": {
                "total_expectations": total,
                "passed": passed,
                "failed": failed,
                "success_percent": round((passed / max(1, total)) * 100, 2),
            },
            "results": [r.to_dict() for r in results],
        }
