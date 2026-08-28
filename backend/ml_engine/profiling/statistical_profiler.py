"""
ModelForge AI - ML Engine: Statistical Dataset Profiler
Calculates mean, std, median, min, max, skewness, kurtosis, quantiles,
Pearson correlation matrix, and histogram bins per column.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


class StatisticalProfiler:
    """Enterprise dataset statistical profiling engine."""

    @staticmethod
    def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute full statistical profile of a pandas DataFrame."""
        column_stats = {}
        histograms = {}
        missingness = {}

        total_rows = len(df)

        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            null_pct = float((null_count / max(1, total_rows)) * 100.0)
            distinct_count = int(series.nunique(dropna=True))

            missingness[col] = {
                "null_count": null_count,
                "null_pct": round(null_pct, 2),
                "distinct_count": distinct_count,
            }

            if pd.api.types.is_numeric_dtype(series):
                clean_s = series.dropna().astype(float)
                if len(clean_s) > 0:
                    mean_val = float(clean_s.mean())
                    std_val = float(clean_s.std()) if len(clean_s) > 1 else 0.0
                    median_val = float(clean_s.median())
                    min_val = float(clean_s.min())
                    max_val = float(clean_s.max())
                    skew_val = float(stats.skew(clean_s)) if len(clean_s) > 2 else 0.0
                    kurt_val = float(stats.kurtosis(clean_s)) if len(clean_s) > 2 else 0.0

                    q25 = float(clean_s.quantile(0.25))
                    q50 = float(clean_s.quantile(0.50))
                    q75 = float(clean_s.quantile(0.75))

                    column_stats[col] = {
                        "type": "numeric",
                        "count": int(len(clean_s)),
                        "null_count": null_count,
                        "null_pct": round(null_pct, 2),
                        "distinct_count": distinct_count,
                        "mean": round(mean_val, 4),
                        "std": round(std_val, 4),
                        "median": round(median_val, 4),
                        "min": round(min_val, 4),
                        "max": round(max_val, 4),
                        "skewness": round(skew_val, 4),
                        "kurtosis": round(kurt_val, 4),
                        "quantiles": {"q25": round(q25, 4), "q50": round(q50, 4), "q75": round(q75, 4)},
                    }

                    # Histogram bins (10 bins)
                    counts, bin_edges = np.histogram(clean_s, bins=10)
                    histograms[col] = [
                        {
                            "bin_start": round(float(bin_edges[i]), 2),
                            "bin_end": round(float(bin_edges[i + 1]), 2),
                            "count": int(counts[i]),
                        }
                        for i in range(len(counts))
                    ]
                else:
                    column_stats[col] = {"type": "numeric", "count": 0, "null_count": null_count}
            else:
                # Categorical or String
                clean_s = series.dropna().astype(str)
                top_values = clean_s.value_counts().head(10).to_dict()
                column_stats[col] = {
                    "type": "categorical",
                    "count": int(len(clean_s)),
                    "null_count": null_count,
                    "null_pct": round(null_pct, 2),
                    "distinct_count": distinct_count,
                    "top_values": top_values,
                }
                # Frequency histogram
                histograms[col] = [
                    {"category": str(k), "count": int(v)}
                    for k, v in top_values.items()
                ]

        # Compute Pearson Correlation Matrix for numeric columns
        correlations = {}
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr(method="pearson").fillna(0.0)
            correlations = {
                "columns": list(corr_matrix.columns),
                "matrix": corr_matrix.round(3).values.tolist(),
            }

        return {
            "column_stats": column_stats,
            "correlations": correlations,
            "histograms": histograms,
            "missingness_matrix": missingness,
        }
