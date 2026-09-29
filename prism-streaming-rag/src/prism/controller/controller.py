"""
PRISM Adaptive Retrieval Controller (Phase 8) — CORE RESEARCH CONTRIBUTION

The controller takes the current QueryState and decides:
  WAIT      — not enough information yet
  RETRIEVE  — retrieval is warranted
  SUPPRESS  — retrieval not needed (reformulation, repeat, low novelty)

Decision logic (transparent, deterministic):
  1. Compute all signals (novelty, stability, completeness, evidence gap)
  2. Compute retrieval utility via the configured policy
  3. Apply gate conditions
  4. Output an explicit ControllerDecision with full audit trail

Every decision is logged to the TelemetryBus with:
  - timestamp
  - transcript state
  - action
  - all scores
  - reason

This is the primary research contribution of the PRISM project.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .query_state import QueryState, RetrievalAction
from .utility import RetrievalUtilityEstimator, UtilityResult, UtilitySignals

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Controller Decision Output
# ──────────────────────────────────────────────────────────────

@dataclass
class ControllerDecision:
    """
    Explicit, auditable decision from the Adaptive Retrieval Controller.

    This is the primary output of the controller at each transcript chunk.
    All fields are logged to telemetry.
    """

    action: RetrievalAction
    reason: str

    # Signal scores
    novelty_score: float = 0.0
    intent_stability: float = 0.0
    query_completeness: float = 0.0
    evidence_gap: float = 1.0
    retrieval_utility: float = 0.0

    # Utility breakdown
    utility_breakdown: Dict[str, float] = field(default_factory=dict)

    # Context
    chunk_index: int = 0
    session_id: str = ""
    transcript_text: str = ""
    timestamp: float = field(default_factory=time.time)
    retrieval_id: Optional[str] = None   # Set when action == RETRIEVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "novelty_score": round(self.novelty_score, 4),
            "intent_stability": round(self.intent_stability, 4),
            "query_completeness": round(self.query_completeness, 4),
            "evidence_gap": round(self.evidence_gap, 4),
            "retrieval_utility": round(self.retrieval_utility, 4),
            "utility_breakdown": {k: round(v, 4) for k, v in self.utility_breakdown.items()},
            "chunk_index": self.chunk_index,
            "session_id": self.session_id,
            "transcript_text": self.transcript_text,
            "timestamp": self.timestamp,
            "retrieval_id": self.retrieval_id,
        }


# ──────────────────────────────────────────────────────────────
# Controller Configuration
# ──────────────────────────────────────────────────────────────

@dataclass
class ControllerConfig:
    retrieve_threshold: float = 0.55
    suppress_threshold: float = 0.20
    min_completeness_to_retrieve: float = 0.40
    min_intent_stability_to_retrieve: float = 0.50
    min_retrieval_interval_s: float = 2.0
    evidence_sufficient_threshold: float = 0.70

    @classmethod
    def from_config(cls, config) -> "ControllerConfig":
        """Build from PRISMConfig.controller."""
        c = config.controller
        return cls(
            retrieve_threshold=c.retrieve_threshold,
            suppress_threshold=c.suppress_threshold,
            min_completeness_to_retrieve=c.min_completeness_to_retrieve,
            min_intent_stability_to_retrieve=c.min_intent_stability_to_retrieve,
            min_retrieval_interval_s=c.min_retrieval_interval_s,
            evidence_sufficient_threshold=c.evidence.sufficient_coverage_threshold,
        )


# ──────────────────────────────────────────────────────────────
# Suppress Pattern Detector
# ──────────────────────────────────────────────────────────────

SUPPRESS_PATTERNS = [
    "repeat",
    "summarize again",
    "bullet point",
    "rephrase",
    "say again",
    "restate",
    "format",
    "in a table",
    "shorter",
    "longer",
    "simpler",
    "translate",
]


def is_reformulation_request(text: str) -> bool:
    """
    Detect if the user is asking for reformulation rather than new information.
    These requests should be SUPPRESSED (no retrieval needed).
    """
    text_lower = text.lower()
    return any(pat in text_lower for pat in SUPPRESS_PATTERNS)


# ──────────────────────────────────────────────────────────────
# Adaptive Retrieval Controller
# ──────────────────────────────────────────────────────────────

class AdaptiveRetrievalController:
    """
    Core research contribution: the Adaptive Retrieval Controller.

    On every transcript chunk, it:
      1. Checks for suppress patterns (reformulation requests)
      2. Evaluates evidence sufficiency
      3. Applies gate conditions on completeness and stability
      4. Computes retrieval utility
      5. Makes WAIT / RETRIEVE / SUPPRESS decision

    All decisions include full signal scores for interpretability.

    Args:
        utility_estimator: Computes utility score from signals
        config: Controller thresholds
    """

    def __init__(
        self,
        utility_estimator: Optional[RetrievalUtilityEstimator] = None,
        config: Optional[ControllerConfig] = None,
    ) -> None:
        self.utility_estimator = utility_estimator or RetrievalUtilityEstimator()
        self.config = config or ControllerConfig()

    def decide(
        self,
        state: QueryState,
        evidence_coverage: float = 0.0,
        retrieval_cost_estimate: float = 0.1,
    ) -> ControllerDecision:
        """
        Make a WAIT / RETRIEVE / SUPPRESS decision.

        Args:
            state: Current QueryState (has novelty, stability, completeness, etc.)
            evidence_coverage: How well current evidence covers the query [0, 1]
            retrieval_cost_estimate: Normalized retrieval cost [0, 1]

        Returns:
            ControllerDecision with full audit trail
        """
        chunk_idx = state.chunk_index
        text = state.current_text
        novelty = state.novelty_score
        stability = state.intent_stability
        completeness = state.completeness
        seconds_since = state.seconds_since_last_retrieval()

        # ── Gate 1: Suppress patterns ──────────────────────────
        if is_reformulation_request(text):
            return self._make_decision(
                action=RetrievalAction.SUPPRESS,
                reason="reformulation_request_detected",
                state=state,
                evidence_gap=1.0 - evidence_coverage,
                utility=0.0,
                utility_result=None,
            )

        # ── Gate 2: Evidence already sufficient ────────────────
        evidence_gap = max(0.0, 1.0 - evidence_coverage)
        if (
            evidence_coverage >= self.config.evidence_sufficient_threshold
            and novelty < 0.15
        ):
            return self._make_decision(
                action=RetrievalAction.SUPPRESS,
                reason="evidence_sufficient_no_novelty",
                state=state,
                evidence_gap=evidence_gap,
                utility=0.0,
                utility_result=None,
            )

        # ── Compute utility ────────────────────────────────────
        signals = UtilitySignals(
            novelty_score=novelty,
            intent_stability=stability,
            completeness=completeness,
            evidence_gap=evidence_gap,
            retrieval_cost=retrieval_cost_estimate,
            seconds_since_last_retrieval=seconds_since,
            chunk_index=chunk_idx,
        )
        utility_result = self.utility_estimator.compute(signals)
        utility = utility_result.utility

        # ── Gate 3: Minimum completeness ───────────────────────
        if completeness < self.config.min_completeness_to_retrieve:
            if utility >= self.config.retrieve_threshold:
                # Override: very high utility even with low completeness
                action = RetrievalAction.RETRIEVE
                reason = "high_utility_overrides_completeness_gate"
            else:
                action = RetrievalAction.WAIT
                reason = f"completeness={completeness:.2f} below threshold={self.config.min_completeness_to_retrieve:.2f}"
            return self._make_decision(
                action=action,
                reason=reason,
                state=state,
                evidence_gap=evidence_gap,
                utility=utility,
                utility_result=utility_result,
            )

        # ── Gate 4: Minimum intent stability ───────────────────
        if stability < self.config.min_intent_stability_to_retrieve:
            if utility >= self.config.retrieve_threshold + 0.10:
                action = RetrievalAction.RETRIEVE
                reason = "very_high_utility_overrides_stability_gate"
            else:
                action = RetrievalAction.WAIT
                reason = f"intent_stability={stability:.2f} below threshold={self.config.min_intent_stability_to_retrieve:.2f}"
            return self._make_decision(
                action=action,
                reason=reason,
                state=state,
                evidence_gap=evidence_gap,
                utility=utility,
                utility_result=utility_result,
            )

        # ── Gate 5: Utility threshold ──────────────────────────
        if utility >= self.config.retrieve_threshold:
            action = RetrievalAction.RETRIEVE
            reason = f"utility={utility:.3f} >= retrieve_threshold={self.config.retrieve_threshold:.2f}"
        elif utility <= self.config.suppress_threshold:
            action = RetrievalAction.SUPPRESS
            reason = f"utility={utility:.3f} <= suppress_threshold={self.config.suppress_threshold:.2f}"
        else:
            action = RetrievalAction.WAIT
            reason = (
                f"utility={utility:.3f} between suppress ({self.config.suppress_threshold:.2f}) "
                f"and retrieve ({self.config.retrieve_threshold:.2f})"
            )

        return self._make_decision(
            action=action,
            reason=reason,
            state=state,
            evidence_gap=evidence_gap,
            utility=utility,
            utility_result=utility_result,
        )

    def _make_decision(
        self,
        action: RetrievalAction,
        reason: str,
        state: QueryState,
        evidence_gap: float,
        utility: float,
        utility_result: Optional[UtilityResult],
    ) -> ControllerDecision:
        retrieval_id = str(uuid.uuid4()) if action == RetrievalAction.RETRIEVE else None

        decision = ControllerDecision(
            action=action,
            reason=reason,
            novelty_score=state.novelty_score,
            intent_stability=state.intent_stability,
            query_completeness=state.completeness,
            evidence_gap=evidence_gap,
            retrieval_utility=utility,
            utility_breakdown=utility_result.breakdown if utility_result else {},
            chunk_index=state.chunk_index,
            session_id=state.session_id,
            transcript_text=state.current_text,
            retrieval_id=retrieval_id,
        )

        logger.info(
            "CONTROLLER [chunk=%d] action=%s reason=%s utility=%.3f",
            state.chunk_index,
            action.value,
            reason,
            utility,
        )

        return decision
