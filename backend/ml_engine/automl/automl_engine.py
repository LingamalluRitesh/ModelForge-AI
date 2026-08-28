"""
ModelForge AI - ML Engine: Complete AutoML Pipeline Orchestrator
Automates problem detection, preprocessing, algorithm selection, HPO, ranking, and leaderboard generation.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import numpy as np
import pandas as pd
from app.core.logging import logger
from app.ml_engine.preprocessing.imputers import AdvancedImputer, AdvancedScaler
from app.ml_engine.preprocessing.encoders import AdvancedCategoricalEncoder
from app.ml_engine.algorithms.classification import get_classifier
from app.ml_engine.algorithms.regression import get_regressor
from app.ml_engine.evaluation.classification_evaluator import ClassificationEvaluator, RegressionEvaluator
from app.ml_engine.optimization.optuna_optimizer import OptunaHPOEngine


class AutoMLPipeline:
    """Enterprise AutoML engine executing end-to-end automated machine learning."""

    def __init__(
        self,
        target_column: str,
        problem_type: Optional[str] = None,
        optimization_metric: str = "f1",
        max_trials_per_algo: int = 5,
        time_limit_seconds: int = 1800,
        random_state: int = 42,
    ):
        self.target_column = target_column
        self.problem_type = problem_type
        self.optimization_metric = optimization_metric
        self.max_trials_per_algo = max_trials_per_algo
        self.time_limit_seconds = time_limit_seconds
        self.random_state = random_state

        # Preprocessing artifacts
        self.imputer = AdvancedImputer()
        self.encoder = AdvancedCategoricalEncoder()
        self.scaler = AdvancedScaler()

        self.leaderboard: List[Dict[str, Any]] = []
        self.best_model: Any = None
        self.best_algorithm_name: Optional[str] = None
        self.best_score: float = 0.0

    def detect_problem_type(self, y: pd.Series) -> str:
        """Infer whether problem is binary_classification, multiclass_classification, or regression."""
        if self.problem_type:
            return self.problem_type

        if pd.api.types.is_numeric_dtype(y):
            unique_count = y.nunique()
            if unique_count == 2:
                return "binary_classification"
            elif 2 < unique_count <= 20 and pd.api.types.is_integer_dtype(y):
                return "multiclass_classification"
            else:
                return "regression"
        else:
            return "binary_classification" if y.nunique() == 2 else "multiclass_classification"

    def fit_transform_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Clean, impute, encode, and scale features."""
        X_df = df.drop(columns=[self.target_column]).copy()
        y_series = df[self.target_column].copy()

        # Fit preprocessing pipeline
        X_imp = self.imputer.fit_transform(X_df)
        X_enc = self.encoder.fit_transform(X_imp)
        X_scaled = self.scaler.fit_transform(X_enc)

        feature_names = list(X_scaled.columns)
        return X_scaled.values, y_series.values, feature_names

    def run_automl(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute full AutoML model exploration, tuning, and leaderboard construction."""
        start_time = time.time()
        self.problem_type = self.detect_problem_type(df[self.target_column])

        is_cls = "classification" in self.problem_type
        if is_cls and self.optimization_metric not in ["f1", "accuracy", "roc_auc"]:
            self.optimization_metric = "f1"
        elif not is_cls and self.optimization_metric not in ["r2", "rmse", "mae"]:
            self.optimization_metric = "r2"

        X_arr, y_arr, feature_names = self.fit_transform_features(df)

        # Train/test split (80/20)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr, test_size=0.2, random_state=self.random_state,
            stratify=y_arr if is_cls and len(np.unique(y_arr)) > 1 else None,
        )

        candidate_algorithms = [
            "random_forest",
            "xgboost",
            "lightgbm",
            "logistic_regression" if is_cls else "linear_regression",
            "decision_tree" if is_cls else "gradient_boosting_regression",
        ]

        leaderboard_entries = []

        for algo_name in candidate_algorithms:
            if (time.time() - start_time) > self.time_limit_seconds:
                logger.info(f"AutoML reached time limit ({self.time_limit_seconds}s). Stopping exploration.")
                break

            try:
                algo_start = time.time()
                # Run Optuna HPO
                hpo_engine = OptunaHPOEngine(
                    algorithm_name=algo_name,
                    problem_type="classification" if is_cls else "regression",
                    optimization_metric=self.optimization_metric,
                    n_trials=self.max_trials_per_algo,
                    timeout_seconds=300,
                    random_state=self.random_state,
                )
                hpo_results = hpo_engine.optimize(X_train, y_train)
                best_params = hpo_results["best_params"]

                # Fit final model with best hyperparameters
                if is_cls:
                    model = get_classifier(algo_name, **best_params)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    probs = model.predict_proba(X_test)
                    metrics = ClassificationEvaluator.evaluate(y_test, preds, probs)
                    score = metrics[self.optimization_metric] if self.optimization_metric in metrics else metrics["accuracy"]
                else:
                    model = get_regressor(algo_name, **best_params)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    metrics = RegressionEvaluator.evaluate(y_test, preds)
                    score = metrics[self.optimization_metric] if self.optimization_metric in metrics else metrics["r2"]

                algo_duration = time.time() - algo_start

                entry = {
                    "algorithm": algo_name,
                    "score": round(float(score), 4),
                    "primary_metric": self.optimization_metric,
                    "metrics": metrics,
                    "hyperparameters": best_params,
                    "duration_seconds": round(algo_duration, 2),
                    "model_obj": model,
                }
                leaderboard_entries.append(entry)

            except Exception as e:
                logger.warning(f"AutoML trial failed for algorithm '{algo_name}': {e}")

        # Rank leaderboard descending by score
        is_higher_better = self.optimization_metric in ["f1", "accuracy", "roc_auc", "r2"]
        leaderboard_entries.sort(key=lambda x: x["score"], reverse=is_higher_better)

        if leaderboard_entries:
            champion = leaderboard_entries[0]
            self.best_model = champion["model_obj"]
            self.best_algorithm_name = champion["algorithm"]
            self.best_score = champion["score"]

        # Clean serializable leaderboard
        clean_leaderboard = [
            {k: v for k, v in entry.items() if k != "model_obj"}
            for entry in leaderboard_entries
        ]
        self.leaderboard = clean_leaderboard

        return {
            "best_algorithm": self.best_algorithm_name,
            "best_score": self.best_score,
            "optimization_metric": self.optimization_metric,
            "problem_type": self.problem_type,
            "leaderboard": clean_leaderboard,
            "total_duration_seconds": round(time.time() - start_time, 2),
        }
