"""
PRISM Telemetry Bus

Every pipeline event emits a structured TelemetryEvent.
Events are:
  - written to a JSONL log file
  - stored in SQLite (optional)
  - broadcast to connected WebSocket clients (optional)

This is the primary observability mechanism for G6.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Event Types
# ──────────────────────────────────────────────────────────────

class EventType(str, Enum):
    # Streaming
    TRANSCRIPT_CHUNK = "transcript_chunk"

    # Controller
    RETRIEVAL_DECISION = "retrieval_decision"
    NOVELTY_COMPUTED = "novelty_computed"
    INTENT_CLASSIFIED = "intent_classified"
    UTILITY_COMPUTED = "utility_computed"

    # Retrieval
    RETRIEVAL_START = "retrieval_start"
    RETRIEVAL_COMPLETE = "retrieval_complete"
    SUBQUERY_START = "subquery_start"
    SUBQUERY_COMPLETE = "subquery_complete"
    BM25_RESULT = "bm25_result"
    DENSE_RESULT = "dense_result"
    RRF_RESULT = "rrf_result"
    RERANK_RESULT = "rerank_result"

    # Evidence
    EVIDENCE_ADDED = "evidence_added"
    EVIDENCE_EXPIRED = "evidence_expired"
    EVIDENCE_COVERAGE = "evidence_coverage"

    # Generation
    GENERATION_START = "generation_start"
    GENERATION_COMPLETE = "generation_complete"
    ANSWER_VERSION = "answer_version"

    # Refinement
    CONSTRAINT_DETECTED = "constraint_detected"
    CLAIM_IMPACT = "claim_impact"
    DELTA_RETRIEVAL_START = "delta_retrieval_start"
    DELTA_RETRIEVAL_COMPLETE = "delta_retrieval_complete"
    ANSWER_REFINED = "answer_refined"

    # Grounding
    CITATION_VALIDATED = "citation_validated"
    GROUNDING_CHECK = "grounding_check"
    UNSUPPORTED_CLAIM = "unsupported_claim"

    # Session
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Error
    ERROR = "error"


# ──────────────────────────────────────────────────────────────
# Event Dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class TelemetryEvent:
    event_type: EventType
    session_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            **self.payload,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ──────────────────────────────────────────────────────────────
# Telemetry Bus
# ──────────────────────────────────────────────────────────────

class TelemetryBus:
    """
    Central event bus for PRISM telemetry.

    Usage:
        bus = TelemetryBus(log_dir="results/telemetry")
        bus.emit(TelemetryEvent(EventType.RETRIEVAL_DECISION, session_id="...", payload={...}))

    Subscribers (WebSocket handlers, DB writers) register via add_subscriber().
    """

    def __init__(
        self,
        log_dir: str = "results/telemetry",
        log_to_file: bool = True,
        log_level: str = "INFO",
    ) -> None:
        self._log_to_file = log_to_file
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._subscribers: List[Callable[[TelemetryEvent], Any]] = []
        self._async_subscribers: List[Callable[[TelemetryEvent], Any]] = []
        self._event_log: List[TelemetryEvent] = []

        # Per-session JSONL file handles
        self._file_handles: Dict[str, Any] = {}

        logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

    def add_subscriber(self, fn: Callable[[TelemetryEvent], None]) -> None:
        """Register a synchronous subscriber."""
        self._subscribers.append(fn)

    def add_async_subscriber(self, fn: Callable[[TelemetryEvent], Any]) -> None:
        """Register an async subscriber (coroutine function)."""
        self._async_subscribers.append(fn)

    def emit(self, event: TelemetryEvent) -> None:
        """Emit an event synchronously."""
        self._event_log.append(event)

        # Log to structured logger
        logger.debug("TELEMETRY [%s] %s", event.event_type.value, event.to_dict())

        # Write to JSONL file
        if self._log_to_file:
            self._write_to_file(event)

        # Notify sync subscribers
        for fn in self._subscribers:
            try:
                fn(event)
            except Exception as exc:
                logger.warning("Telemetry subscriber error: %s", exc)

    async def emit_async(self, event: TelemetryEvent) -> None:
        """Emit an event and notify async subscribers."""
        self.emit(event)
        for fn in self._async_subscribers:
            try:
                await fn(event)
            except Exception as exc:
                logger.warning("Async telemetry subscriber error: %s", exc)

    def _write_to_file(self, event: TelemetryEvent) -> None:
        session_id = event.session_id
        if session_id not in self._file_handles:
            log_path = self._log_dir / f"session_{session_id}.jsonl"
            self._file_handles[session_id] = open(log_path, "a", encoding="utf-8")

        fh = self._file_handles[session_id]
        fh.write(event.to_json() + "\n")
        fh.flush()

    def get_session_events(
        self, session_id: str, event_type: Optional[EventType] = None
    ) -> List[TelemetryEvent]:
        """Return all events for a session (optionally filtered by type)."""
        events = [e for e in self._event_log if e.session_id == session_id]
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        return events

    def get_decision_log(self, session_id: str) -> List[Dict[str, Any]]:
        """Return controller decision log for a session (for frontend display)."""
        events = self.get_session_events(session_id, EventType.RETRIEVAL_DECISION)
        return [e.to_dict() for e in events]

    def close(self) -> None:
        for fh in self._file_handles.values():
            try:
                fh.close()
            except Exception:
                pass
        self._file_handles.clear()

    def __del__(self) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────
# Global bus instance (module-level singleton)
# ──────────────────────────────────────────────────────────────

_bus: Optional[TelemetryBus] = None


def get_telemetry_bus(log_dir: str = "results/telemetry") -> TelemetryBus:
    global _bus
    if _bus is None:
        _bus = TelemetryBus(log_dir=log_dir)
    return _bus
