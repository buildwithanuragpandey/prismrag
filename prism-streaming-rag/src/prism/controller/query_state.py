"""
PRISM Query State Tracker (Phase 4)

Maintains evolving query state across streaming transcript chunks.

The QueryState captures:
  - The current query text
  - Query completeness score (0-1)
  - Detected intents and their evolution
  - History of all states
  - Retrieval history

This is the stateful backbone that the Adaptive Controller reads from.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────

class QueryCompletenessLevel(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    COMPLETE = "complete"


class RetrievalAction(str, Enum):
    WAIT = "WAIT"
    RETRIEVE = "RETRIEVE"
    SUPPRESS = "SUPPRESS"


# ──────────────────────────────────────────────────────────────
# Intent
# ──────────────────────────────────────────────────────────────

@dataclass
class DetectedIntent:
    """A single detected intent within the query."""
    label: str                          # e.g. "venue_selection", "cancellation_policy"
    description: str = ""              # Natural language description
    confidence: float = 0.0            # 0-1
    constraints: List[str] = field(default_factory=list)  # e.g. ["capacity >= 30"]
    sub_query: str = ""                # Retrieval sub-query for this intent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "description": self.description,
            "confidence": self.confidence,
            "constraints": self.constraints,
            "sub_query": self.sub_query,
        }


# ──────────────────────────────────────────────────────────────
# Per-chunk State Snapshot
# ──────────────────────────────────────────────────────────────

@dataclass
class ChunkStateSnapshot:
    """
    Immutable snapshot of query state at a particular transcript chunk.
    Stored in history for session-level analysis.
    """
    chunk_index: int
    text: str
    delta_text: str
    timestamp: float
    intents: List[DetectedIntent]
    completeness: float                    # 0-1
    completeness_level: QueryCompletenessLevel
    novelty_score: float                   # 0-1 (how novel vs. previous chunk)
    intent_stability: float                # 0-1
    retrieval_utility: float               # 0-1
    action: RetrievalAction
    action_reason: str = ""
    retrieval_id: Optional[str] = None     # Set if RETRIEVE was executed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "text": self.text,
            "delta_text": self.delta_text,
            "timestamp": self.timestamp,
            "intents": [i.to_dict() for i in self.intents],
            "completeness": round(self.completeness, 4),
            "completeness_level": self.completeness_level.value,
            "novelty_score": round(self.novelty_score, 4),
            "intent_stability": round(self.intent_stability, 4),
            "retrieval_utility": round(self.retrieval_utility, 4),
            "action": self.action.value,
            "action_reason": self.action_reason,
            "retrieval_id": self.retrieval_id,
        }


# ──────────────────────────────────────────────────────────────
# Live Query State
# ──────────────────────────────────────────────────────────────

@dataclass
class QueryState:
    """
    Live, mutable query state for a session.
    Updated on every transcript chunk.

    This is the central state object read by:
      - SemanticNoveltyDetector
      - IntentStabilityTracker
      - RetrievalUtilityEstimator
      - AdaptiveRetrievalController
    """

    session_id: str

    # Current text state
    current_text: str = ""
    previous_text: str = ""
    chunk_index: int = -1

    # Signals (updated per chunk)
    novelty_score: float = 0.0
    intent_stability: float = 0.0
    completeness: float = 0.0
    completeness_level: QueryCompletenessLevel = QueryCompletenessLevel.UNKNOWN
    retrieval_utility: float = 0.0

    # Intent tracking
    current_intents: List[DetectedIntent] = field(default_factory=list)
    previous_intents: List[DetectedIntent] = field(default_factory=list)
    intent_history: List[List[str]] = field(default_factory=list)  # per-chunk intent labels

    # Retrieval tracking
    last_retrieval_time: float = 0.0
    last_retrieval_text: str = ""
    total_retrievals: int = 0
    retrieval_history: List[Dict[str, Any]] = field(default_factory=list)

    # History
    history: List[ChunkStateSnapshot] = field(default_factory=list)

    def record_retrieval(self, retrieval_id: str, query: str) -> None:
        """Record that a retrieval was performed."""
        self.last_retrieval_time = time.time()
        self.last_retrieval_text = query
        self.total_retrievals += 1
        self.retrieval_history.append({
            "retrieval_id": retrieval_id,
            "query": query,
            "chunk_index": self.chunk_index,
            "timestamp": self.last_retrieval_time,
        })

    def seconds_since_last_retrieval(self) -> float:
        if self.last_retrieval_time == 0.0:
            return float("inf")
        return time.time() - self.last_retrieval_time

    def get_intent_labels(self) -> List[str]:
        return [i.label for i in self.current_intents]

    def snapshot(
        self,
        action: RetrievalAction,
        action_reason: str = "",
        retrieval_id: Optional[str] = None,
    ) -> ChunkStateSnapshot:
        """Take an immutable snapshot of current state."""
        snap = ChunkStateSnapshot(
            chunk_index=self.chunk_index,
            text=self.current_text,
            delta_text=self.current_text[len(self.previous_text):],
            timestamp=time.time(),
            intents=list(self.current_intents),
            completeness=self.completeness,
            completeness_level=self.completeness_level,
            novelty_score=self.novelty_score,
            intent_stability=self.intent_stability,
            retrieval_utility=self.retrieval_utility,
            action=action,
            action_reason=action_reason,
            retrieval_id=retrieval_id,
        )
        self.history.append(snap)
        return snap

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "chunk_index": self.chunk_index,
            "current_text": self.current_text,
            "completeness": round(self.completeness, 4),
            "completeness_level": self.completeness_level.value,
            "novelty_score": round(self.novelty_score, 4),
            "intent_stability": round(self.intent_stability, 4),
            "retrieval_utility": round(self.retrieval_utility, 4),
            "intents": [i.to_dict() for i in self.current_intents],
            "total_retrievals": self.total_retrievals,
            "seconds_since_last_retrieval": round(
                self.seconds_since_last_retrieval(), 2
            ),
        }


# ──────────────────────────────────────────────────────────────
# Query State Tracker
# ──────────────────────────────────────────────────────────────

class QueryStateTracker:
    """
    Manages QueryState objects per session.
    On each transcript chunk, updates the QueryState fields.

    Note: The QueryStateTracker itself does NOT compute novelty/stability/utility —
    those are computed by their respective components and injected back via update().
    This keeps the tracker as a clean stateful container.
    """

    def __init__(self) -> None:
        self._states: Dict[str, QueryState] = {}

    def get_or_create(self, session_id: str) -> QueryState:
        """Get existing state or create a new one."""
        if session_id not in self._states:
            self._states[session_id] = QueryState(session_id=session_id)
        return self._states[session_id]

    def update_from_chunk(
        self,
        session_id: str,
        chunk_index: int,
        current_text: str,
    ) -> QueryState:
        """
        Called on each new transcript chunk.
        Updates text fields; signal scores are set separately by each component.
        """
        state = self.get_or_create(session_id)
        state.previous_text = state.current_text
        state.previous_intents = list(state.current_intents)
        state.current_text = current_text
        state.chunk_index = chunk_index
        return state

    def apply_signals(
        self,
        state: QueryState,
        novelty_score: float,
        intent_stability: float,
        completeness: float,
        retrieval_utility: float,
        intents: List[DetectedIntent],
    ) -> None:
        """Inject computed signal scores into the state."""
        state.novelty_score = novelty_score
        state.intent_stability = intent_stability
        state.completeness = completeness
        state.retrieval_utility = retrieval_utility

        # Completeness level bucketing
        if completeness >= 0.85:
            state.completeness_level = QueryCompletenessLevel.COMPLETE
        elif completeness >= 0.65:
            state.completeness_level = QueryCompletenessLevel.HIGH
        elif completeness >= 0.40:
            state.completeness_level = QueryCompletenessLevel.MEDIUM
        elif completeness > 0.10:
            state.completeness_level = QueryCompletenessLevel.LOW
        else:
            state.completeness_level = QueryCompletenessLevel.UNKNOWN

        state.current_intents = intents
        state.intent_history.append([i.label for i in intents])

    def all_sessions(self) -> List[str]:
        return list(self._states.keys())

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        state = self._states.get(session_id)
        if state is None:
            return []
        return [snap.to_dict() for snap in state.history]
