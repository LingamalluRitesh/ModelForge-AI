"""
ModelForge AI - Data Pipeline: Complex Event Processing (CEP) Non-Deterministic Finite Automaton Engine
Implements Agrawal et al. Efficient Pattern Matching over Event Streams (SASE / Cayuga)
evaluating sequence patterns with Kleene-plus operators, predicates, and sliding temporal windows.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time


class CEPPatternState:
    def __init__(self, state_id: str, is_terminal: bool = False):
        self.state_id = state_id
        self.is_terminal = is_terminal
        # transition_predicate_fn: (event, match_history) -> bool
        self.transitions: List[Tuple[Callable[[Dict[str, Any], List[Dict[str, Any]]], bool], str]] = []

    def add_transition(self, predicate: Callable[[Dict[str, Any], List[Dict[str, Any]]], bool], target_state_id: str):
        self.transitions.append((predicate, target_state_id))


class PartialMatchInstance:
    def __init__(self, current_state_id: str, events: List[Dict[str, Any]]):
        self.state_id = current_state_id
        self.events = events
        self.start_timestamp = events[0].get("timestamp", time.time()) if events else time.time()


class CEPAutomatonEngine:
    """Evaluates incoming real-time telemetry streams against multi-step complex behavioral DAG patterns."""

    def __init__(self, states: Dict[str, CEPPatternState], start_state_id: str, max_window_seconds: float = 300.0):
        self.states = states
        self.start_state_id = start_state_id
        self.max_window = max_window_seconds
        self.active_instances: List[PartialMatchInstance] = []

    def process_event(self, event: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Processes one incoming streaming event and returns matched complex event patterns."""
        now = event.get("timestamp", time.time())
        # Evict expired instances
        self.active_instances = [
            inst for inst in self.active_instances if (now - inst.start_timestamp) <= self.max_window
        ]

        completed_matches: List[List[Dict[str, Any]]] = []
        new_instances: List[PartialMatchInstance] = []

        # 1. Spawn candidate start instance if start state matches
        start_st = self.states[self.start_state_id]
        for pred, next_st in start_st.transitions:
            if pred(event, []):
                inst = PartialMatchInstance(next_st, [event])
                if self.states[next_st].is_terminal:
                    completed_matches.append(inst.events)
                else:
                    new_instances.append(inst)

        # 2. Advance existing active match instances
        for inst in self.active_instances:
            curr_st = self.states[inst.state_id]
            for pred, next_st in curr_st.transitions:
                if pred(event, inst.events):
                    updated_events = inst.events + [event]
                    if self.states[next_st].is_terminal:
                        completed_matches.append(updated_events)
                    else:
                        new_instances.append(PartialMatchInstance(next_st, updated_events))

        self.active_instances.extend(new_instances)
        return completed_matches
