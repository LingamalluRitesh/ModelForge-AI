"""
ModelForge AI - ML Engine: Metapath2Vec Random Walk Schema Generator
Implements Dong, Chawla, & Swami metapath2vec: Scalable Representation Learning for Heterogeneous Networks
generating formal schema-constrained random walk trajectories (e.g., Author -> Paper -> Venue -> Paper -> Author).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class MetapathRandomWalkEngine:
    """Generates heterogeneous random walks obeying user-specified metapath schemas."""
    def __init__(self, graph_adj: Dict[str, Dict[int, List[int]]]):
        # relation_type -> (source_id -> list of target_ids)
        self.adj = graph_adj

    def generate_walk(self, start_node_id: int, metapath_schema: List[str], walk_length: int = 20) -> List[Tuple[str, int]]:
        """
        metapath_schema: list of relation types like ['writes_paper', 'published_in_venue', 'venue_contains_paper', 'paper_written_by']
        """
        walk: List[Tuple[str, int]] = []
        curr_node = start_node_id
        schema_len = len(metapath_schema)

        for step in range(walk_length):
            rel_type = metapath_schema[step % schema_len]
            neighbors = self.adj.get(rel_type, {}).get(curr_node, [])

            if not neighbors:
                break

            next_node = int(np.random.choice(neighbors))
            walk.append((rel_type, next_node))
            curr_node = next_node

        return walk
