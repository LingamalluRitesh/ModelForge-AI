"""
ModelForge AI - ML Engine: Hyperparameter Optimization (Optuna)
Implements Bayesian Tree-Structured Parzen Estimator (TPE) optimization,
cross-validation evaluation, and trial pruning.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, KFold
from app.core.logging import logger
from ml_engine.algorithms.classification import get_classifier
from ml_engine.algorithms.regression import get_regressor
from ml_engine.evaluation.classification_evaluator import ClassificationEvaluator, RegressionEvaluator

# Silence Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaHPOEngine:
    """Enterprise Bayesian HPO Engine using Optuna TPE sampler."""

    def __init__(
        self,
        algorithm_name: str,
        problem_type: str = "classification",
        optimization_metric: str = "f1",
        n_trials: int = 20,
        timeout_seconds: int = 1800,
        n_splits: int = 3,
        random_state: int = 42,
    ):
        self.algorithm_name = algorithm_name
        self.problem_type = problem_type
        self.optimization_metric = optimization_metric
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.n_splits = n_splits
        self.random_state = random_state

        self.study: Optional[optuna.Study] = None
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = 0.0
        self.trials_history: List[Dict[str, Any]] = []

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample algorithm-specific hyperparameters from search spaces."""
        algo = self.algorithm_name.lower().replace("-", "_")

        if "random_forest" in algo:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 20, 150, step=10),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": self.random_state,
            }
        elif "xgboost" in algo:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 20, 150, step=10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": self.random_state,
            }
        elif "lightgbm" in algo:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 20, 150, step=10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": self.random_state,
            }
        elif "logistic_regression" in algo:
            return {
                "C": trial.suggest_float("C", 0.01, 100.0, log=True),
                "max_iter": 1000,
                "random_state": self.random_state,
            }
        elif "decision_tree" in algo:
            return {
                "max_depth": trial.suggest_int("max_depth", 2, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": self.random_state,
            }
        else:
            return {"random_state": self.random_state}

    def optimize(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """Execute Bayesian Optimization with K-Fold Cross-Validation."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        direction = "maximize" if self.optimization_metric in ["f1", "accuracy", "roc_auc", "r2"] else "minimize"
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

        self.study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

        is_cls = self.problem_type == "classification"
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state) if is_cls else KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_hyperparameters(trial)
            scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y if is_cls else None)):
                X_train_f, X_val_f = X[train_idx], X[val_idx]
                y_train_f, y_val_f = y[train_idx], y[val_idx]

                if is_cls:
                    model = get_classifier(self.algorithm_name, **params)
                    model.fit(X_train_f, y_train_f)
                    preds = model.predict(X_val_f)
                    probs = model.predict_proba(X_val_f)
                    metrics = ClassificationEvaluator.evaluate(y_val_f, preds, probs)
                    score = metrics.get(self.optimization_metric, metrics["accuracy"])
                else:
                    model = get_regressor(self.algorithm_name, **params)
                    model.fit(X_train_f, y_train_f)
                    preds = model.predict(X_val_f)
                    metrics = RegressionEvaluator.evaluate(y_val_f, preds)
                    score = metrics.get(self.optimization_metric, metrics["rmse"])

                scores.append(score)

                # Report intermediate score for pruning
                trial.report(float(np.mean(scores)), fold_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            mean_score = float(np.mean(scores))
            return mean_score

        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout_seconds)

        self.best_params = self.study.best_params
        self.best_score = float(self.study.best_value)

        # Record trial histories
        for t in self.study.trials:
            self.trials_history.append({
                "trial_number": t.number,
                "params": t.params,
                "score": float(t.value) if t.value is not None else 0.0,
                "state": t.state.name,
                "duration_seconds": float(t.duration.total_seconds()) if t.duration else 0.0,
            })

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "trials_count": len(self.study.trials),
            "trials_history": self.trials_history,
        }
