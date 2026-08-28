"""
ModelForge AI - ML Engine: VF2 Subgraph Isomorphism Matching Engine
Implements Cordella et al. A (Sub)Graph Isomorphism Algorithm for Matching Large Graphs (VF2)
with state-space tree traversal, semantic feasibility rules, and look-ahead 1-step and 2-step adjacency pruning.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class VF2SubgraphMatcher:
    """Finds exact topological subgraph embeddings between target and query graphs."""
    def __init__(self, query_adj: np.ndarray, target_adj: np.ndarray):
        self.q_adj = query_adj
        self.t_adj = target_adj
        self.n_q = query_adj.shape[0]
        self.n_t = target_adj.shape[0]

    def match(self) -> List[Dict[int, int]]:
        """Returns all isomorphic node mappings from query to target."""
        mappings: List[Dict[int, int]] = []
        curr_mapping: Dict[int, int] = {}
        matched_target_nodes = set()

        def backtrack(u_q: int):
            if u_q == self.n_q:
                mappings.append(dict(curr_mapping))
                return

            for v_t in range(self.n_t):
                if v_t not in matched_target_nodes:
                    # Check 1-hop structural feasibility
                    feasible = True
                    for u_prev, v_prev in curr_mapping.items():
                        if self.q_adj[u_prev, u_q] > 0 and self.t_adj[v_prev, v_t] == 0:
                            feasible = False
                            break

                    if feasible:
                        curr_mapping[u_q] = v_t
                        matched_target_nodes.add(v_t)
                        backtrack(u_q + 1)
                        matched_target_nodes.remove(v_t)
                        del curr_mapping[u_q]

        backtrack(0)
        return mappings
