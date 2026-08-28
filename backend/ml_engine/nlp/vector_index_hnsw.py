"""
ModelForge AI - ML Engine: Hierarchical Navigable Small World (HNSW) Vector Search Index
Implements Malkov & Yashunin HNSW graph index for sub-millisecond approximate nearest neighbor (ANN)
vector similarity search over high-dimensional embedding feature vectors.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import heapq
import numpy as np


class HNSWNode:
    def __init__(self, node_id: int, vector: np.ndarray, level: int):
        self.node_id = node_id
        self.vector = np.asarray(vector, dtype=np.float32)
        self.level = level
        # Neighbor lists per level: level_idx -> list of neighbor node IDs
        self.friends: Dict[int, List[int]] = {l: [] for l in range(level + 1)}


class HNSWIndex:
    """Multi-layer Hierarchical Navigable Small World Graph Vector Index."""

    def __init__(
        self,
        dim: int = 128,
        metric: str = "cosine",  # cosine, l2, inner_product
        M: int = 16,             # Max outgoing connections per node per layer
        M0: int = 32,            # Max connections in ground layer 0
        ef_construction: int = 100,  # Size of dynamic candidate list during insertion
        ef_search: int = 50,         # Size of candidate list during query
        mL: float = 1.0 / np.log(16), # Normalization factor for level generation
    ):
        self.dim = dim
        self.metric = metric
        self.M = M
        self.M0 = M0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = mL

        self.nodes: Dict[int, HNSWNode] = {}
        self.enter_point_id: Optional[int] = None
        self.max_level: int = -1

    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if self.metric == "cosine":
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 1.0
            return float(1.0 - np.dot(v1, v2) / (norm1 * norm2))
        elif self.metric == "inner_product":
            return float(-np.dot(v1, v2))
        else:  # l2
            return float(np.sum((v1 - v2) ** 2))

    def _random_level(self) -> int:
        """Assign random level with exponentially decaying probability."""
        r = np.random.uniform(0, 1)
        return int(-np.log(r) * self.mL)

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        level: int,
    ) -> List[Tuple[float, int]]:
        """Greedy best-first search within a single graph layer."""
        v = set(entry_points)
        candidates: List[Tuple[float, int]] = []  # Min-heap of candidates (dist, node_id)
        w: List[Tuple[float, int]] = []           # Max-heap of nearest found (neg_dist, node_id)

        for ep in entry_points:
            dist = self._distance(query, self.nodes[ep].vector)
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(w, (-dist, ep))

        while len(candidates) > 0:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_dist = -w[0][0]

            if c_dist > furthest_dist:
                break

            for neighbor_id in self.nodes[c_id].friends.get(level, []):
                if neighbor_id not in v:
                    v.add(neighbor_id)
                    n_dist = self._distance(query, self.nodes[neighbor_id].vector)
                    furthest_dist = -w[0][0]

                    if n_dist < furthest_dist or len(w) < ef:
                        heapq.heappush(candidates, (n_dist, neighbor_id))
                        heapq.heappush(w, (-n_dist, neighbor_id))
                        if len(w) > ef:
                            heapq.heappop(w)

        # Return sorted list (dist, node_id)
        res = [(-neg_d, node_id) for neg_d, node_id in w]
        res.sort(key=lambda x: x[0])
        return res

    def insert(self, node_id: int, vector: np.ndarray):
        """Insert high-dimensional vector into hierarchical HNSW graph."""
        vec = np.asarray(vector, dtype=np.float32)
        node_level = self._random_level()
        new_node = HNSWNode(node_id, vec, node_level)
        self.nodes[node_id] = new_node

        if self.enter_point_id is None:
            self.enter_point_id = node_id
            self.max_level = node_level
            return

        curr_obj = self.enter_point_id
        curr_dist = self._distance(vec, self.nodes[curr_obj].vector)

        # 1. Greedy routing down to node's level
        for l in range(self.max_level, node_level, -1):
            changed = True
            while changed:
                changed = False
                for neighbor_id in self.nodes[curr_obj].friends.get(l, []):
                    d = self._distance(vec, self.nodes[neighbor_id].vector)
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor_id
                        changed = True

        # 2. Connect neighbors from bottom levels up to node_level
        ep_list = [curr_obj]
        for l in range(min(self.max_level, node_level), -1, -1):
            neighbors = self._search_layer(vec, ep_list, self.ef_construction, l)
            max_m = self.M0 if l == 0 else self.M
            selected_neighbors = [nid for _, nid in neighbors[:max_m]]

            new_node.friends[l] = selected_neighbors
            for n_id in selected_neighbors:
                self.nodes[n_id].friends[l].append(node_id)
                # Prune if exceeding capacity
                if len(self.nodes[n_id].friends[l]) > max_m:
                    n_vec = self.nodes[n_id].vector
                    ranked = [(self._distance(n_vec, self.nodes[f_id].vector), f_id) for f_id in self.nodes[n_id].friends[l]]
                    ranked.sort(key=lambda x: x[0])
                    self.nodes[n_id].friends[l] = [nid for _, nid in ranked[:max_m]]

            ep_list = [nid for _, nid in neighbors]

        if node_level > self.max_level:
            self.max_level = node_level
            self.enter_point_id = node_id

    def query(self, vector: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search top-k nearest neighbors in sub-millisecond graph walk."""
        if self.enter_point_id is None:
            return []

        vec = np.asarray(vector, dtype=np.float32)
        curr_obj = self.enter_point_id
        curr_dist = self._distance(vec, self.nodes[curr_obj].vector)

        for l in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor_id in self.nodes[curr_obj].friends.get(l, []):
                    d = self._distance(vec, self.nodes[neighbor_id].vector)
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor_id
                        changed = True

        results = self._search_layer(vec, [curr_obj], max(k, self.ef_search), 0)
        return [{"node_id": nid, "distance": round(dist, 5)} for dist, nid in results[:k]]
