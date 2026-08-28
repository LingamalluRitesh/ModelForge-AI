"""
ModelForge AI - ML Engine: Distributed Tracing & OpenTelemetry Collector
Tracks end-to-end inference request lifecycle spans, model prediction latency breakdown,
feature extraction overhead, and correlation trace IDs across microservices.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import time
import uuid
import threading
from contextlib import contextmanager


class Span:
    """Represents a single timed unit of work within a distributed transaction trace."""

    def __init__(self, name: str, trace_id: str, parent_id: Optional[str] = None):
        self.span_id = uuid.uuid4().hex[:16]
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.name = name
        self.start_time = time.perf_counter()
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0.0
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.status: str = "OK"

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def add_event(self, event_name: str, payload: Optional[Dict[str, Any]] = None):
        self.events.append({
            "name": event_name,
            "timestamp": time.time(),
            "payload": payload or {},
        })

    def finish(self, status: str = "OK"):
        self.end_time = time.perf_counter()
        self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 3)
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
        }


class Tracer:
    """In-memory OpenTelemetry tracer recording hierarchical execution traces."""

    def __init__(self, service_name: str = "modelforge-inference"):
        self.service_name = service_name
        self._completed_spans: List[Span] = []
        self._lock = threading.Lock()

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        t_id = trace_id or uuid.uuid4().hex
        span = Span(name=name, trace_id=t_id, parent_id=parent_id)
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)

        try:
            yield span
            span.finish(status="OK")
        except Exception as e:
            span.set_attribute("error.message", str(e))
            span.finish(status="ERROR")
            raise
        finally:
            with self._lock:
                self._completed_spans.append(span)
                if len(self._completed_spans) > 5000:
                    self._completed_spans = self._completed_spans[-2500:]

    def get_recent_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return [s.to_dict() for s in self._completed_spans[-limit:]]


tracer = Tracer()
