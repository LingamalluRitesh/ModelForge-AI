"""
ModelForge AI - Visual DAG Pipeline Orchestrator & Execution Engine
Performs topological sorting, dependency resolution, parallel node dispatch, and error isolation.
"""

from typing import Any, Dict, List, Optional, Set
from collections import defaultdict, deque
import asyncio
import time
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.exceptions import ValidationException
from app.core.logging import logger


class DAGExecutionEngine:
    """Orchestrates multi-node DAG workflows with cycle detection and parallel execution branches."""

    @staticmethod
    def validate_and_topological_sort(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
        """Verify graph is acyclic and return valid execution order."""
        node_ids = {n["id"] for n in nodes}
        adj_list = defaultdict(list)
        in_degree = {nid: 0 for nid in node_ids}

        for edge in edges:
            src, dst = edge["source"], edge["target"]
            if src in node_ids and dst in node_ids:
                adj_list[src].append(dst)
                in_degree[dst] += 1

        # Kahn's algorithm for topological sorting
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        sorted_order = []

        while queue:
            curr = queue.popleft()
            sorted_order.append(curr)

            for neighbor in adj_list[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_order) != len(node_ids):
            raise ValidationException("Cycle detected in DAG pipeline definition!")

        return sorted_order

    @staticmethod
    async def execute_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate asynchronous execution of a pipeline step."""
        node_type = node.get("type", "task")
        name = node.get("name", node.get("id"))
        t0 = time.perf_counter()

        # Simulate task duration based on node type
        if node_type == "ingest":
            await asyncio.sleep(0.05)
        elif node_type == "train":
            await asyncio.sleep(0.1)
        elif node_type == "evaluate":
            await asyncio.sleep(0.05)
        else:
            await asyncio.sleep(0.02)

        dur = time.perf_counter() - t0
        return {
            "node_id": node["id"],
            "name": name,
            "status": "COMPLETED",
            "duration_seconds": round(dur, 3),
            "output_keys": [f"output_{node['id']}"],
        }
