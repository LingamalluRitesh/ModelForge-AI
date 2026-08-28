"""
ModelForge AI - ML Engine: Data Quality Engine
Validates missing values, duplicates, outliers, range violations, regex constraints,
type consistencies, and computes composite Data Quality Scores (0 - 100%).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import re
import numpy as np
import pandas as pd


class DataQualityEngine:
    """Enterprise Data Quality & Validation Engine."""

    def __init__(self, custom_rules: Optional[List[Dict[str, Any]]] = None):
        self.custom_rules = custom_rules or []

    def evaluate_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive automated quality checks on the dataset."""
        total_rows = len(df)
        total_cols = len(df.columns)

        if total_rows == 0 or total_cols == 0:
            return {
                "quality_score": 0.0,
                "passed_rules": 0,
                "failed_rules": 1,
                "total_checks": 1,
                "missing_value_percentage": 100.0,
                "duplicate_rows_count": 0,
                "outlier_count": 0,
                "anomalies": {"empty_dataset": True},
                "rule_results": [],
                "summary": "Dataset is empty.",
            }

        rule_results = []
        passed_rules = 0
        failed_rules = 0
        penalty = 0.0

        # 1. Missing Values Analysis
        total_cells = total_rows * total_cols
        null_cells = int(df.isnull().sum().sum())
        missing_pct = float((null_cells / total_cells) * 100.0)

        missing_rule_passed = missing_pct < 10.0
        if missing_rule_passed:
            passed_rules += 1
        else:
            failed_rules += 1
            penalty += min(25.0, missing_pct)

        rule_results.append({
            "rule": "Missing Values Threshold (< 10%)",
            "passed": missing_rule_passed,
            "details": f"Overall missing rate is {missing_pct:.2f}% ({null_cells} cells)",
            "severity": "error" if missing_pct >= 20.0 else "warning",
        })

        # 2. Duplicate Rows Analysis
        duplicate_count = int(df.duplicated().sum())
        duplicate_pct = float((duplicate_count / total_rows) * 100.0)
        dup_passed = duplicate_pct < 2.0
        if dup_passed:
            passed_rules += 1
        else:
            failed_rules += 1
            penalty += min(15.0, duplicate_pct * 2.0)

        rule_results.append({
            "rule": "Duplicate Rows Check (< 2%)",
            "passed": dup_passed,
            "details": f"Duplicate rows: {duplicate_count} ({duplicate_pct:.2f}%)",
            "severity": "warning",
        })

        # 3. Constant Columns Check
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        const_passed = len(constant_cols) == 0
        if const_passed:
            passed_rules += 1
        else:
            failed_rules += 1
            penalty += len(constant_cols) * 5.0

        rule_results.append({
            "rule": "Constant Columns Check (zero variance)",
            "passed": const_passed,
            "details": f"Found constant columns: {constant_cols}" if constant_cols else "No constant columns found.",
            "severity": "warning",
        })

        # 4. Outliers via IQR
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_total = 0
        outlier_details = {}

        for col in numeric_cols:
            col_series = df[col].dropna()
            if len(col_series) > 10:
                q25 = col_series.quantile(0.25)
                q75 = col_series.quantile(0.75)
                iqr = q75 - q25
                lower = q25 - 2.5 * iqr
                upper = q75 + 2.5 * iqr
                col_outliers = int(((col_series < lower) | (col_series > upper)).sum())
                if col_outliers > 0:
                    outlier_total += col_outliers
                    outlier_details[col] = col_outliers

        outlier_pct = float((outlier_total / max(1, len(numeric_cols) * total_rows)) * 100.0) if len(numeric_cols) > 0 else 0.0
        outlier_passed = outlier_pct < 5.0
        if outlier_passed:
            passed_rules += 1
        else:
            failed_rules += 1
            penalty += min(10.0, outlier_pct)

        rule_results.append({
            "rule": "Extreme Outliers Check (< 5%)",
            "passed": outlier_passed,
            "details": f"Extreme outliers detected: {outlier_total} ({outlier_pct:.2f}%)",
            "severity": "warning",
        })

        # 5. Evaluate Custom Configured Rules
        for rule in self.custom_rules:
            col_name = rule.get("column_name")
            rule_type = rule.get("rule_type")
            params = rule.get("params", {})

            if col_name not in df.columns:
                failed_rules += 1
                rule_results.append({
                    "rule": f"Custom Rule: {col_name} {rule_type}",
                    "passed": False,
                    "details": f"Column '{col_name}' does not exist in dataset.",
                    "severity": "error",
                })
                penalty += 10.0
                continue

            passed = True
            msg = "Rule satisfied."

            if rule_type == "not_null":
                null_cnt = int(df[col_name].isnull().sum())
                passed = null_cnt == 0
                msg = f"{null_cnt} null values found in '{col_name}'" if not passed else msg

            elif rule_type == "range":
                min_v = params.get("min_val")
                max_v = params.get("max_val")
                num_s = pd.to_numeric(df[col_name], errors="coerce").dropna()
                violations = 0
                if min_v is not None:
                    violations += int((num_s < min_v).sum())
                if max_v is not None:
                    violations += int((num_s > max_v).sum())
                passed = violations == 0
                msg = f"{violations} values outside [{min_v}, {max_v}] in '{col_name}'" if not passed else msg

            elif rule_type == "unique":
                dups = int(df[col_name].duplicated().sum())
                passed = dups == 0
                msg = f"{dups} non-unique values in '{col_name}'" if not passed else msg

            elif rule_type == "regex":
                pattern = params.get("pattern", ".*")
                compiled = re.compile(pattern)
                invalid = int(df[col_name].dropna().astype(str).map(lambda v: bool(compiled.match(v)) is False).sum())
                passed = invalid == 0
                msg = f"{invalid} values do not match regex '{pattern}'" if not passed else msg

            if passed:
                passed_rules += 1
            else:
                failed_rules += 1
                penalty += 10.0

            rule_results.append({
                "rule": f"Custom Rule on '{col_name}': {rule_type}",
                "passed": passed,
                "details": msg,
                "severity": rule.get("severity", "error"),
            })

        # Calculate final quality score (0.0 to 100.0)
        final_score = max(0.0, min(100.0, 100.0 - penalty))

        return {
            "quality_score": round(final_score, 1),
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "total_checks": passed_rules + failed_rules,
            "missing_value_percentage": round(missing_pct, 2),
            "duplicate_rows_count": duplicate_count,
            "outlier_count": outlier_total,
            "anomalies": {
                "constant_columns": constant_cols,
                "outliers_by_column": outlier_details,
            },
            "rule_results": rule_results,
            "summary": f"Data Quality Score: {final_score:.1f}% ({passed_rules} of {passed_rules + failed_rules} checks passed).",
        }
