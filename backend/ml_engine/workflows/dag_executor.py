"""
ModelForge AI - ML Engine: Asynchronous DAG Workflow & Pipeline Orchestrator
Implements Kahn's Topological Sorting Algorithm, concurrent asynchronous task scheduling,
dependency state resolution, retry exponential backoff, and execution checkpointing.
"""

from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union
import asyncio
import time
import uuid
import logging

logger = logging.getLogger("modelforge.dag_orchestrator")


class TaskStatus(str):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PipelineTask:
    """Individual executable node within an ML pipeline DAG."""

    def __init__(
        self,
        task_id: str,
        name: str,
        handler: Callable[..., Any],
        dependencies: Optional[List[str]] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        self.task_id = task_id
        self.name = name
        self.handler = handler
        self.dependencies: Set[str] = set(dependencies or [])
        self.max_retries = max_retries
        self.retry_delay = retry_delay_seconds

        self.status = TaskStatus.PENDING
        self.output: Any = None
        self.error: Optional[str] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_seconds: float = 0.0

    async def execute(self, inputs: Dict[str, Any]) -> Any:
        self.status = TaskStatus.RUNNING
        self.start_time = time.perf_counter()

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Executing task {self.task_id} ({self.name}), attempt {attempt}/{self.max_retries}")
                if asyncio.iscoroutinefunction(self.handler):
                    self.output = await self.handler(inputs)
                else:
                    self.output = self.handler(inputs)

                self.status = TaskStatus.COMPLETED
                self.end_time = time.perf_counter()
                self.duration_seconds = round(self.end_time - self.start_time, 3)
                return self.output
            except Exception as e:
                logger.warning(f"Task {self.task_id} attempt {attempt} failed: {e}")
                if attempt == self.max_retries:
                    self.status = TaskStatus.FAILED
                    self.error = str(e)
                    self.end_time = time.perf_counter()
                    self.duration_seconds = round(self.end_time - self.start_time, 3)
                    raise
                await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))


class DAGPipelineExecutor:
    """DAG Pipeline Orchestrator executing tasks concurrently as dependencies resolve."""

    def __init__(self, pipeline_id: str, name: str):
        self.pipeline_id = pipeline_id
        self.name = name
        self.tasks: Dict[str, PipelineTask] = {}
        self.context: Dict[str, Any] = {}

    def add_task(self, task: PipelineTask):
        self.tasks[task.task_id] = task

    def _get_topological_order(self) -> List[str]:
        """Kahn's Algorithm for cycle detection and topological ordering."""
        in_degree: Dict[str, int] = {t_id: 0 for t_id in self.tasks}
        graph: Dict[str, List[str]] = {t_id: [] for t_id in self.tasks}

        for t_id, task in self.tasks.items():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    raise ValueError(f"Dependency '{dep}' not found for task '{t_id}'")
                graph[dep].append(t_id)
                in_degree[t_id] += 1

        queue = [t_id for t_id, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.tasks):
            raise ValueError("Cyclic dependency detected in ML pipeline DAG.")

        return order

    async def run(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run all DAG tasks with maximum parallel concurrency."""
        self._get_topological_order()  # Validate acyclic
        self.context = (initial_context or {}).copy()

        completed_tasks: Set[str] = set()
        running_tasks: Dict[str, asyncio.Task] = {}

        t0 = time.perf_counter()

        while len(completed_tasks) < len(self.tasks):
            # Find tasks ready to run
            ready_tasks = [
                task
                for t_id, task in self.tasks.items()
                if task.status == TaskStatus.PENDING
                and task.dependencies.issubset(completed_tasks)
                and t_id not in running_tasks
            ]

            for task in ready_tasks:
                inputs = {dep: self.tasks[dep].output for dep in task.dependencies}
                inputs.update(self.context)
                coro = task.execute(inputs)
                running_tasks[task.task_id] = asyncio.create_task(coro)

            if not running_tasks:
                break

            # Wait for any running task to complete
            done, _ = await asyncio.wait(
                list(running_tasks.values()), return_when=asyncio.FIRST_COMPLETED
            )

            for future in done:
                finished_task_id = [t_id for t_id, t in running_tasks.items() if t == future][0]
                del running_tasks[finished_task_id]

                try:
                    out = future.result()
                    completed_tasks.add(finished_task_id)
                    self.context[f"task_{finished_task_id}_output"] = out
                except Exception as e:
                    logger.error(f"Pipeline {self.pipeline_id} failed on task {finished_task_id}: {e}")
                    raise

        total_duration = round(time.perf_counter() - t0, 3)

        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.name,
            "status": "COMPLETED",
            "total_duration_seconds": total_duration,
            "task_summaries": {
                t_id: {
                    "name": t.name,
                    "status": t.status,
                    "duration": t.duration_seconds,
                    "error": t.error,
                }
                for t_id, t in self.tasks.items()
            },
        }
