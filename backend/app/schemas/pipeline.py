"""
ModelForge AI - Visual DAG Pipeline & Workflow Scheduler Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class DAGNode(BaseModel):
    id: str
    type: str # dataset, validation, feature_eng, train, evaluate, approve, deploy, notify
    name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    position: Dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0})


class DAGEdge(BaseModel):
    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None


class DAGDefinition(BaseModel):
    nodes: List[DAGNode]
    edges: List[DAGEdge]


class MLPipelineCreate(BaseModel):
    name: str
    description: Optional[str] = None
    dag_definition: DAGDefinition
    tags: Optional[List[str]] = None


class MLPipelineUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    dag_definition: Optional[DAGDefinition] = None
    is_active: Optional[bool] = None
    tags: Optional[List[str]] = None


class PipelineNodeExecutionResponse(BaseModel):
    id: str
    node_id: str
    node_type: str
    status: str
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    logs: Optional[str] = None
    duration_seconds: Optional[float] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class PipelineRunResponse(BaseModel):
    id: str
    pipeline_id: str
    run_number: int
    status: str
    trigger_type: str
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    node_executions: List[PipelineNodeExecutionResponse] = []

    model_config = {"from_attributes": True}


class WorkflowScheduleCreate(BaseModel):
    name: str
    cron_expression: str
    is_enabled: bool = True


class WorkflowScheduleResponse(BaseModel):
    id: str
    pipeline_id: str
    name: str
    cron_expression: str
    is_enabled: bool
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MLPipelineResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: Optional[str] = None
    dag_definition: Dict[str, Any]
    is_active: bool
    tags: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime
    latest_run: Optional[PipelineRunResponse] = None
    schedules: List[WorkflowScheduleResponse] = []

    model_config = {"from_attributes": True}
