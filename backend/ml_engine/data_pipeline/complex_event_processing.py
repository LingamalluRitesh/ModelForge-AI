"""
ModelForge AI - ML Engine: Complex Event Processing (CEP) State Automata
Implements Non-Deterministic Finite Automata (NFA) for streaming pattern detection
(e.g., Sequence matching: LoginFailed -> LoginFailed -> LoginSuccess within 60 seconds).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time


class EventPatternStep:
    def __init__(self, step_name: str, predicate_fn: Callable[[Dict[str, Any]], bool]):
        self.step_name = step_name
        self.predicate = predicate_fn


class StreamingCEPAutomata:
    """Non-Deterministic Finite Automata for real-time temporal pattern recognition."""
    def __init__(self, pattern_name: str, steps: List[EventPatternStep], time_window_seconds: float = 60.0):
        self.pattern_name = pattern_name
        self.steps = steps
        self.time_window = time_window_seconds
        self.active_sequences: List[Dict[str, Any]] = []

    def process_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        now = time.time()
        # 1. Purge expired sequences
        self.active_sequences = [
            seq for seq in self.active_sequences if now - seq["start_time"] <= self.time_window
        ]

        matched_pattern = None

        # 2. Advance existing sequences
        for seq in self.active_sequences:
            current_step_idx = seq["current_step"]
            if current_step_idx < len(self.steps):
                step = self.steps[current_step_idx]
                if step.predicate(event):
                    seq["matched_events"].append(event)
                    seq["current_step"] += 1

                    if seq["current_step"] == len(self.steps):
                        matched_pattern = {
                            "pattern_name": self.pattern_name,
                            "matched_events": seq["matched_events"],
                            "duration_seconds": round(now - seq["start_time"], 3),
                        }

        # 3. Check if event initiates a new sequence
        first_step = self.steps[0]
        if first_step.predicate(event):
            self.active_sequences.append({
                "start_time": now,
                "current_step": 1,
                "matched_events": [event],
            })

        return matched_pattern
