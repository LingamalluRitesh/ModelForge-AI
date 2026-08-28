"""
ModelForge AI - Event-Driven Architecture & Message Bus
Handles system events (e.g. DatasetCreated, TrainingCompleted, ModelDeployed, DriftDetected)
with synchronous, asynchronous in-memory listeners and Redis/Kafka dispatchers.
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio
from datetime import datetime, timezone
import json
from pydantic import BaseModel, Field
from app.core.logging import logger


class SystemEventType(str, Enum):
    # Data events
    DATASET_CREATED = "dataset.created"
    DATASET_VALIDATED = "dataset.validated"
    DATASET_PROFILED = "dataset.profiled"
    FEATURE_REGISTERED = "feature.registered"

    # Training & AutoML events
    EXPERIMENT_STARTED = "experiment.started"
    TRAINING_STARTED = "training.started"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_FAILED = "training.failed"
    AUTOML_STARTED = "automl.started"
    AUTOML_COMPLETED = "automl.completed"

    # Governance & Registry events
    MODEL_REGISTERED = "model.registered"
    MODEL_APPROVAL_REQUESTED = "model.approval_requested"
    MODEL_APPROVED = "model.approved"
    MODEL_REJECTED = "model.rejected"

    # Deployment & Serving events
    MODEL_DEPLOYED = "model.deployed"
    DEPLOYMENT_SCALED = "deployment.scaled"
    DEPLOYMENT_ROLLED_BACK = "deployment.rolled_back"
    CANARY_PROMOTED = "deployment.canary_promoted"
    PREDICTION_EXECUTED = "prediction.executed"

    # Monitoring & Drift events
    DRIFT_DETECTED = "drift.detected"
    PERFORMANCE_DEGRADED = "performance.degraded"
    RETRAINING_TRIGGERED = "retraining.triggered"
    RETRAINING_COMPLETED = "retraining.completed"

    # Alert & Security events
    ALERT_TRIGGERED = "alert.triggered"
    SECURITY_EVENT = "security.event"
    AUDIT_EVENT = "audit.event"


class SystemEvent(BaseModel):
    """Standardized event envelope structure."""
    event_id: str
    event_type: SystemEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: Optional[str] = None
    organization_id: Optional[str] = None
    project_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class EventBus:
    """In-process and distributed asynchronous event pub/sub bus."""

    def __init__(self):
        self._subscribers: Dict[SystemEventType, List[Callable]] = {}

    def subscribe(self, event_type: SystemEventType, handler: Callable):
        """Register an async or sync event handler callback."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler '{handler.__name__}' to event '{event_type.value}'")

    async def publish(self, event: SystemEvent):
        """Publish an event to all registered listeners and log to stream."""
        logger.info(
            f"Event Published: [{event.event_type.value}] - Org: {event.organization_id}, "
            f"Project: {event.project_id}, Actor: {event.actor_id}"
        )

        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler '{handler.__name__}' for event '{event.event_type}': {e}")


# Global EventBus instance
event_bus = EventBus()
