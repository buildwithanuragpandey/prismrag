"""
PRISM Baselines (Phase 16)

Implements three comparison baselines for the adaptive controller:

BASELINE 1: ConventionalRAG
  - Wait for complete query, then retrieve once, generate once
  - No streaming awareness

BASELINE 2: NaiveStreamingRAG
  - Retrieve on EVERY transcript chunk
  - No intelligence applied

BASELINE 3: FixedThresholdRAG
  - Retrieve when query completeness exceeds a fixed threshold
  - No novelty/stability signals

These baselines are used in the benchmark evaluation and ablation studies.
Each exposes the same interface as AdaptiveRetrievalController.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from prism.controller.query_state import QueryState, RetrievalAction
from prism.controller.completeness import estimate_completeness


# ──────────────────────────────────────────────────────────────
# Base Interface
# ──────────────────────────────────────────────────────────────

@dataclass
class BaselineDecision:
    action: RetrievalAction
    reason: str
    chunk_index: int
    session_id: str
    retrieval_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "chunk_index": self.chunk_index,
            "session_id": self.session_id,
            "retrieval_id": self.retrieval_id,
            **self.metadata,
        }


class BaselineController(ABC):
    name: str = "baseline"

    @abstractmethod
    def decide(self, state: QueryState, **kwargs) -> BaselineDecision:
        ...

    def reset_session(self, session_id: str) -> None:
        pass


# ──────────────────────────────────────────────────────────────
# Baseline 1: Conventional RAG
# ──────────────────────────────────────────────────────────────

class ConventionalRAG(BaselineController):
    """
    Conventional RAG baseline:
    - WAIT until the last (final) chunk
    - Then RETRIEVE once
    - Never retrieves mid-stream

    This represents the standard non-streaming RAG approach.
    """

    name = "conventional_rag"

    def __init__(self) -> None:
        self._retrieved: set = set()

    def decide(self, state: QueryState, is_final: bool = False, **kwargs) -> BaselineDecision:
        if is_final and state.session_id not in self._retrieved:
            self._retrieved.add(state.session_id)
            return BaselineDecision(
                action=RetrievalAction.RETRIEVE,
                reason="final_chunk_conventional_rag",
                chunk_index=state.chunk_index,
                session_id=state.session_id,
                retrieval_id=str(uuid.uuid4()),
            )
        return BaselineDecision(
            action=RetrievalAction.WAIT,
            reason="waiting_for_final_chunk",
            chunk_index=state.chunk_index,
            session_id=state.session_id,
        )

    def reset_session(self, session_id: str) -> None:
        self._retrieved.discard(session_id)


# ──────────────────────────────────────────────────────────────
# Baseline 2: Naive Streaming RAG
# ──────────────────────────────────────────────────────────────

class NaiveStreamingRAG(BaselineController):
    """
    Naive Streaming RAG baseline:
    - RETRIEVE on every single transcript chunk
    - No intelligence, no suppression, no waiting
    - Represents the worst-case retrieval cost

    This is the upper bound on retrieval calls.
    """

    name = "naive_streaming_rag"

    def decide(self, state: QueryState, **kwargs) -> BaselineDecision:
        return BaselineDecision(
            action=RetrievalAction.RETRIEVE,
            reason="retrieve_every_chunk",
            chunk_index=state.chunk_index,
            session_id=state.session_id,
            retrieval_id=str(uuid.uuid4()),
        )


# ──────────────────────────────────────────────────────────────
# Baseline 3: Fixed Threshold RAG
# ──────────────────────────────────────────────────────────────

class FixedThresholdRAG(BaselineController):
    """
    Fixed Threshold Streaming RAG baseline:
    - RETRIEVE when estimated query completeness exceeds threshold
    - No novelty, stability, or utility signals
    - Configurable threshold (default 0.60)

    This tests whether simple completeness gating alone is competitive.
    """

    name = "fixed_threshold_rag"

    def __init__(
        self,
        completeness_threshold: float = 0.60,
        min_retrieval_interval_s: float = 3.0,
    ) -> None:
        self.threshold = completeness_threshold
        self.min_interval = min_retrieval_interval_s
        self._last_retrieval: Dict[str, float] = {}

    def decide(self, state: QueryState, **kwargs) -> BaselineDecision:
        completeness = estimate_completeness(state.current_text)
        last = self._last_retrieval.get(state.session_id, 0.0)
        seconds_since = time.time() - last

        if completeness >= self.threshold and seconds_since >= self.min_interval:
            self._last_retrieval[state.session_id] = time.time()
            return BaselineDecision(
                action=RetrievalAction.RETRIEVE,
                reason=f"completeness={completeness:.2f} >= threshold={self.threshold}",
                chunk_index=state.chunk_index,
                session_id=state.session_id,
                retrieval_id=str(uuid.uuid4()),
                metadata={"completeness": round(completeness, 4)},
            )

        return BaselineDecision(
            action=RetrievalAction.WAIT,
            reason=f"completeness={completeness:.2f} < threshold={self.threshold}",
            chunk_index=state.chunk_index,
            session_id=state.session_id,
            metadata={"completeness": round(completeness, 4)},
        )

    def reset_session(self, session_id: str) -> None:
        self._last_retrieval.pop(session_id, None)
