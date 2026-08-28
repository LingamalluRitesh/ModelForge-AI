"""
ModelForge AI - ML Engine: Bandwidth-Optimal Ring-AllReduce
Implements Gibiansky / Baidu Ring-AllReduce gradient synchronization algorithm
dividing parameter tensors into $N$ equal chunks across a logical communication ring.
Total data transferred per worker is $2 \frac{N-1}{N} S$, independent of cluster size $N$.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class RingAllReduceSimulator:
    """Simulates exact Ring-AllReduce Scatter-Reduce and Allgather communication phases."""

    def __init__(self, num_nodes: int = 4):
        self.num_nodes = num_nodes

    def allreduce(self, node_tensors: List[np.ndarray]) -> List[np.ndarray]:
        """
        Execute Scatter-Reduce followed by Allgather over $N$ worker nodes.
        node_tensors: list of $N$ arrays, each of shape $(D,)$
        """
        N = self.num_nodes
        D = len(node_tensors[0])
        chunk_size = D // N

        # Split each node's tensor into N chunks
        chunks = [[node_tensors[i][c * chunk_size : (c + 1) * chunk_size].copy() for c in range(N)] for i in range(N)]

        # Phase 1: Scatter-Reduce (N - 1 steps)
        for step in range(N - 1):
            for i in range(N):
                send_chunk_idx = (i - step) % N
                recv_chunk_idx = (i - step - 1) % N
                receiver_node = (i + 1) % N

                # Send chunk from node i to receiver_node and reduce (sum)
                chunks[receiver_node][recv_chunk_idx] += chunks[i][send_chunk_idx]

        # Phase 2: Allgather (N - 1 steps)
        for step in range(N - 1):
            for i in range(N):
                send_chunk_idx = (i - step + 1) % N
                receiver_node = (i + 1) % N

                # Overwrite receiver's chunk with reduced value
                chunks[receiver_node][send_chunk_idx] = chunks[i][send_chunk_idx].copy()

        # Recombine chunks on all nodes
        reduced_tensors = [np.concatenate(chunks[i]) for i in range(N)]
        return reduced_tensors
