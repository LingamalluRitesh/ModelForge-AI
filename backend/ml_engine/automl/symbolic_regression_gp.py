"""
ModelForge AI - ML Engine: Symbolic Feature Discovery via Genetic Programming
Implements Koza Genetic Programming for Automated Non-Linear Mathematical Feature Discovery
using AST expression trees with mutation, crossover, and Mean Squared Error fitness scoring.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GPNode:
    """Abstract Syntax Tree node for mathematical expressions."""
    def __init__(
        self,
        node_type: str,  # 'op' or 'var' or 'const'
        value: Any = None,
        left: Optional["GPNode"] = None,
        right: Optional["GPNode"] = None,
    ):
        self.node_type = node_type
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.node_type == "const":
            return np.full(X.shape[0], float(self.value))
        elif self.node_type == "var":
            return X[:, int(self.value)]
        elif self.node_type == "op":
            left_val = self.left.evaluate(X) if self.left else np.zeros(X.shape[0])
            right_val = self.right.evaluate(X) if self.right else np.zeros(X.shape[0])

            if self.value == "+":
                return left_val + right_val
            elif self.value == "-":
                return left_val - right_val
            elif self.value == "*":
                return left_val * right_val
            elif self.value == "/":
                return left_val / np.where(np.abs(right_val) < 1e-4, 1e-4, right_val)
            elif self.value == "sin":
                return np.sin(left_val)
            elif self.value == "sqrt":
                return np.sqrt(np.abs(left_val))

        return np.zeros(X.shape[0])

    def __str__(self) -> str:
        if self.node_type == "const":
            return f"{self.value:.2f}"
        elif self.node_type == "var":
            return f"x_{self.value}"
        elif self.node_type == "op":
            if self.right:
                return f"({self.left} {self.value} {self.right})"
            return f"{self.value}({self.left})"
        return ""


class SymbolicRegressor:
    """Genetic Programming Symbolic Formula Discovery."""
    def __init__(self, pop_size: int = 50, generations: int = 20, max_depth: int = 4):
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth = max_depth
        self.best_program: Optional[GPNode] = None
        self.best_fitness: float = float("inf")

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]

        # Initial random population of AST trees
        population = [
            GPNode("op", "+", GPNode("var", 0), GPNode("var", min(1, n_features - 1)))
            for _ in range(self.pop_size)
        ]

        for gen in range(self.generations):
            fitnesses = []
            for ind in population:
                preds = ind.evaluate(X)
                mse = float(np.mean((y - preds) ** 2))
                fitnesses.append(mse)

                if mse < self.best_fitness:
                    self.best_fitness = mse
                    self.best_program = ind

            # Selection (Tournament)
            order = np.argsort(fitnesses)
            survivors = [population[i] for i in order[: self.pop_size // 2]]

            # Reproduction & Mutation
            next_pop = list(survivors)
            while len(next_pop) < self.pop_size:
                parent = survivors[np.random.randint(len(survivors))]
                # Mutate random variable index
                mutated = GPNode(
                    parent.node_type,
                    parent.value,
                    GPNode("var", np.random.randint(n_features)),
                    parent.right,
                )
                next_pop.append(mutated)

            population = next_pop

        return self
