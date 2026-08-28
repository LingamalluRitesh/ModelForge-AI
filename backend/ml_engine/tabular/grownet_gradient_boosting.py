"""
ModelForge AI - ML Engine: GrowNet (Gradient Boosted Neural Networks)
Implements Badirli et al. Gradient Boosted Decision Trees with Neural Networks
training shallow 2-layer MLPs as sequential weak learners fitting pseudo-residuals in function space.
$F_m(x) = F_{m-1}(x) + lpha_m f_m(x)$ where $f_m(x) = rg\min_f \sum_{i=1}^n (r_{im} - f(x_i))^2$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GrowNetWeakLearner:
    """Shallow neural network learner fitting pseudo-residual targets."""
    def __init__(self, in_features: int, hidden_dim: int = 32):
        self.in_features = in_features
        std = np.sqrt(2.0 / in_features)
        self.W1 = np.random.normal(0, std, (in_features, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1))
        self.b2 = np.zeros(1)

    def fit(self, X: np.ndarray, residuals: np.ndarray, epochs: int = 15, lr: float = 0.05):
        N = len(X)
        for _ in range(epochs):
            # Forward
            h1 = np.maximum(0, np.dot(X, self.W1) + self.b1)
            pred = np.dot(h1, self.W2) + self.b2

            # Backward MSE
            err = pred - residuals.reshape(-1, 1)
            grad_W2 = np.dot(h1.T, err) / N
            grad_b2 = np.mean(err, axis=0)

            grad_h1 = np.dot(err, self.W2.T) * (h1 > 0)
            grad_W1 = np.dot(X.T, grad_h1) / N
            grad_b1 = np.mean(grad_h1, axis=0)

            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0, np.dot(X, self.W1) + self.b1)
        return (np.dot(h1, self.W2) + self.b2).reshape(-1)


class GrowNetEnsemble:
    """Sequential ensemble of gradient boosted neural weak learners."""
    def __init__(self, in_features: int, num_boost_rounds: int = 10, learning_rate: float = 0.1):
        self.in_features = in_features
        self.M = num_boost_rounds
        self.lr = learning_rate
        self.learners: List[GrowNetWeakLearner] = []
        self.initial_pred_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.initial_pred_ = float(np.mean(y))

        curr_preds = np.full(len(y), self.initial_pred_)

        for m in range(self.M):
            # Compute negative gradient (pseudo-residuals for MSE)
            residuals = y - curr_preds
            learner = GrowNetWeakLearner(self.in_features)
            learner.fit(X, residuals)

            # Update predictions
            update = learner.predict(X)
            curr_preds += self.lr * update
            self.learners.append(learner)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        preds = np.full(len(X), self.initial_pred_)
        for learner in self.learners:
            preds += self.lr * learner.predict(X)
        return preds
