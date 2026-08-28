"""
ModelForge AI - ML Engine: Multi-Objective NSGA-II Evolutionary Hyperparameter Optimizer
Implements Deb et al. Non-dominated Sorting Genetic Algorithm II (NSGA-II)
with Crowding Distance Assignment, Simulated Binary Crossover (SBX), and Polynomial Mutation.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class Individual:
    """Individual candidate solution in multi-objective hyperparameter search space."""

    def __init__(self, genes: np.ndarray):
        self.genes = np.asarray(genes, dtype=np.float64)
        self.objectives: np.ndarray = np.array([])  # Multi-objective fitness values (to minimize)
        self.rank: int = 0  # Pareto non-domination rank
        self.crowding_distance: float = 0.0
        self.domination_count: int = 0  # Number of individuals that dominate this one ($n_p$)
        self.dominated_solutions: List["Individual"] = []  # Solutions dominated by this one ($S_p$)

    def dominates(self, other: "Individual") -> bool:
        """Pareto domination check: $p \prec q$ iff $\forall i, f_i(p) \le f_i(q)$ and $\exists j, f_j(p) < f_j(q)$."""
        return bool(
            np.all(self.objectives <= other.objectives)
            and np.any(self.objectives < other.objectives)
        )


class NSGA2Optimizer:
    """Multi-Objective Hyperparameter Optimization balancing accuracy, latency, and model size."""

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        objective_funcs: List[Callable[[np.ndarray], float]],
        pop_size: int = 50,
        n_generations: int = 30,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
    ):
        self.bounds = np.array(bounds)
        self.objective_funcs = objective_funcs
        self.n_objectives = len(objective_funcs)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.n_dims = len(bounds)
        self.pareto_front_: List[Individual] = []

    def _evaluate(self, ind: Individual):
        ind.objectives = np.array([f(ind.genes) for f in self.objective_funcs])

    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """Deb's fast non-dominated sorting algorithm with $O(M N^2)$ complexity."""
        fronts: List[List[Individual]] = [[]]

        for p in population:
            p.dominated_solutions = []
            p.domination_count = 0

            for q in population:
                if p.dominates(q):
                    p.dominated_solutions.append(q)
                elif q.dominates(p):
                    p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 1
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front: List[Individual] = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 2
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def _calculate_crowding_distance(self, front: List[Individual]):
        """Assign crowding distance density estimator to maintain diversity."""
        l = len(front)
        if l == 0:
            return

        for ind in front:
            ind.crowding_distance = 0.0

        for m in range(self.n_objectives):
            front.sort(key=lambda ind: ind.objectives[m])
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            f_min = front[0].objectives[m]
            f_max = front[-1].objectives[m]
            diff = max(1e-6, f_max - f_min)

            for i in range(1, l - 1):
                front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / diff

    def _simulated_binary_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Deb & Agrawal Simulated Binary Crossover (SBX)."""
        child1_genes = parent1.genes.copy()
        child2_genes = parent2.genes.copy()

        if np.random.rand() < self.crossover_rate:
            for i in range(self.n_dims):
                if np.random.rand() <= 0.5:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2.0 * u) ** (1.0 / (self.eta_c + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta_c + 1.0))

                    c1 = 0.5 * ((1.0 + beta) * parent1.genes[i] + (1.0 - beta) * parent2.genes[i])
                    c2 = 0.5 * ((1.0 - beta) * parent1.genes[i] + (1.0 + beta) * parent2.genes[i])

                    child1_genes[i] = np.clip(c1, self.bounds[i, 0], self.bounds[i, 1])
                    child2_genes[i] = np.clip(c2, self.bounds[i, 0], self.bounds[i, 1])

        return Individual(child1_genes), Individual(child2_genes)

    def _polynomial_mutation(self, ind: Individual):
        """Polynomial mutation operator."""
        for i in range(self.n_dims):
            if np.random.rand() < self.mutation_rate:
                y = ind.genes[i]
                yl, yu = self.bounds[i, 0], self.bounds[i, 1]
                delta1 = (y - yl) / max(1e-6, yu - yl)
                delta2 = (yu - y) / max(1e-6, yu - yl)

                u = np.random.rand()
                if u <= 0.5:
                    deltaq = ((2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1) ** (self.eta_m + 1.0)) ** (1.0 / (self.eta_m + 1.0))) - 1.0
                else:
                    deltaq = 1.0 - ((2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2) ** (self.eta_m + 1.0)) ** (1.0 / (self.eta_m + 1.0)))

                y = y + deltaq * (yu - yl)
                ind.genes[i] = np.clip(y, yl, yu)

    def optimize(self) -> List[Dict[str, Any]]:
        """Run evolutionary multi-objective optimization loop."""
        # 1. Initialize random population
        population: List[Individual] = []
        for _ in range(self.pop_size):
            genes = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            ind = Individual(genes)
            self._evaluate(ind)
            population.append(ind)

        # 2. Main generation loop
        for gen in range(self.n_generations):
            # Generate offspring
            offspring: List[Individual] = []
            while len(offspring) < self.pop_size:
                # Binary tournament selection
                p1, p2 = np.random.choice(population, 2, replace=False)
                p3, p4 = np.random.choice(population, 2, replace=False)

                parent_a = p1 if (p1.rank < p2.rank or (p1.rank == p2.rank and p1.crowding_distance > p2.crowding_distance)) else p2
                parent_b = p3 if (p3.rank < p4.rank or (p3.rank == p4.rank and p3.crowding_distance > p4.crowding_distance)) else p4

                c1, c2 = self._simulated_binary_crossover(parent_a, parent_b)
                self._polynomial_mutation(c1)
                self._polynomial_mutation(c2)
                self._evaluate(c1)
                self._evaluate(c2)
                offspring.extend([c1, c2])

            # Combine population & offspring: size 2N
            combined = population + offspring[: self.pop_size]
            fronts = self._fast_non_dominated_sort(combined)

            # Construct next generation
            next_pop: List[Individual] = []
            for front in fronts:
                self._calculate_crowding_distance(front)
                if len(next_pop) + len(front) <= self.pop_size:
                    next_pop.extend(front)
                else:
                    # Sort by crowding distance descending and fill remaining
                    front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                    remaining = self.pop_size - len(next_pop)
                    next_pop.extend(front[:remaining])
                    break

            population = next_pop

        # Return Rank 1 Pareto front
        final_fronts = self._fast_non_dominated_sort(population)
        self.pareto_front_ = final_fronts[0]

        return [
            {
                "genes": ind.genes.tolist(),
                "objectives": ind.objectives.tolist(),
                "rank": ind.rank,
                "crowding_distance": ind.crowding_distance,
            }
            for ind in self.pareto_front_
        ]
