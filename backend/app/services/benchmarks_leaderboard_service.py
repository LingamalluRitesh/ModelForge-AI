"""
ModelForge AI - Benchmarks Leaderboard Service
Aggregates model performance, F1, ROC-AUC, latency, and memory footprint across candidate runs.
"""

from typing import Any, Dict, List, Optional


class BenchmarksLeaderboardService:
    @staticmethod
    def get_project_leaderboard(project_id: str) -> List[Dict[str, Any]]:
        return [
            {
                "model_name": "XGBoost-Hist-v2.4",
                "algorithm": "xgboost_hist",
                "accuracy": 0.938,
                "f1_score": 0.884,
                "auc_roc": 0.942,
                "p99_latency_ms": 2.1,
                "memory_footprint_mb": 18.4,
                "rank": 1,
            },
            {
                "model_name": "TabNet-Attentive-v1.8",
                "algorithm": "tabnet",
                "accuracy": 0.921,
                "f1_score": 0.865,
                "auc_roc": 0.928,
                "p99_latency_ms": 4.8,
                "memory_footprint_mb": 42.1,
                "rank": 2,
            },
            {
                "model_name": "NODE-Oblivious-v1.2",
                "algorithm": "node_trees",
                "accuracy": 0.910,
                "f1_score": 0.851,
                "auc_roc": 0.915,
                "p99_latency_ms": 5.2,
                "memory_footprint_mb": 36.8,
                "rank": 3,
            },
            {
                "model_name": "RandomForest-Scratch-v2",
                "algorithm": "random_forest",
                "accuracy": 0.892,
                "f1_score": 0.832,
                "auc_roc": 0.898,
                "p99_latency_ms": 8.4,
                "memory_footprint_mb": 64.0,
                "rank": 4,
            },
        ]
