"""
PRISM Retrieval Utility Estimator (Phase 7)

Combines multiple signals into a single retrieval utility score.
This score determines whether retrieval is worth performing NOW.

Formula (configurable weights):
  utility = w1 * novelty
           + w2 * intent_stability
           + w3 * completeness
           + w4 * evidence_gap
           - w5 * retrieval_cost

All weights are configurable. This is a transparent policy baseline.
We do NOT claim this formula is optimal — it is the research starting point.

A modular interface (UtilityPolicy) is provided so that a learned policy
can be swapped in later for comparison.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Signal Bundle
# ──────────────────────────────────────────────────────────────

@dataclass
class UtilitySignals:
    """All input signals to the utility estimator."""
    novelty_score: float = 0.0          # [0, 1] semantic novelty
    intent_stability: float = 0.0       # [0, 1] intent consistency
    completeness: float = 0.0           # [0, 1] query completeness
    evidence_gap: float = 1.0           # [0, 1] how much evidence is missing
    retrieval_cost: float = 0.1         # [0, 1] normalized cost estimate
    seconds_since_last_retrieval: float = float("inf")
    chunk_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "novelty_score": round(self.novelty_score, 4),
            "intent_stability": round(self.intent_stability, 4),
            "completeness": round(self.completeness, 4),
            "evidence_gap": round(self.evidence_gap, 4),
            "retrieval_cost": round(self.retrieval_cost, 4),
            "seconds_since_last_retrieval": round(self.seconds_since_last_retrieval, 2),
            "chunk_index": self.chunk_index,
        }


@dataclass
class UtilityResult:
    """Output of the utility estimator."""
    utility: float                      # Final utility score [0, 1]
    signals: UtilitySignals
    breakdown: Dict[str, float]         # Contribution of each signal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "utility": round(self.utility, 4),
            "signals": self.signals.to_dict(),
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
        }


# ──────────────────────────────────────────────────────────────
# Abstract Policy Interface (for learned policies later)
# ──────────────────────────────────────────────────────────────

class UtilityPolicy(ABC):
    """Abstract interface for retrieval utility policies."""

    @abstractmethod
    def compute(self, signals: UtilitySignals) -> UtilityResult:
        """Compute utility score from signal bundle."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ──────────────────────────────────────────────────────────────
# Weighted Linear Policy (default)
# ──────────────────────────────────────────────────────────────

class WeightedLinearPolicy(UtilityPolicy):
    """
    Transparent weighted linear combination of signals.

    utility = w1 * novelty
             + w2 * intent_stability
             + w3 * completeness
             + w4 * evidence_gap
             - w5 * retrieval_cost

    All weights configurable. Result clipped to [0, 1].

    This is the PRISM research baseline policy.
    It is intentionally simple and interpretable.
    """

    def __init__(
        self,
        w_novelty: float = 0.30,
        w_intent_stability: float = 0.25,
        w_completeness: float = 0.20,
        w_evidence_gap: float = 0.20,
        w_retrieval_cost: float = 0.05,
        min_retrieval_interval_s: float = 2.0,
    ) -> None:
        self.w_novelty = w_novelty
        self.w_intent_stability = w_intent_stability
        self.w_completeness = w_completeness
        self.w_evidence_gap = w_evidence_gap
        self.w_retrieval_cost = w_retrieval_cost
        self.min_retrieval_interval_s = min_retrieval_interval_s

    @property
    def name(self) -> str:
        return "WeightedLinearPolicy"

    def compute(self, signals: UtilitySignals) -> UtilityResult:
        """
        Compute utility score.

        Time-based cost adjustment: if we retrieved very recently,
        add a cost penalty to discourage rapid re-retrieval.
        """
        # Time-based cost: recent retrieval increases perceived cost
        time_penalty = 0.0
        if signals.seconds_since_last_retrieval < self.min_retrieval_interval_s:
            # Linear penalty from 0 → 1 as seconds_since_last → 0
            time_penalty = 1.0 - (
                signals.seconds_since_last_retrieval / self.min_retrieval_interval_s
            )

        adjusted_cost = min(1.0, signals.retrieval_cost + time_penalty * 0.5)

        # Weighted sum
        positive = (
            self.w_novelty * signals.novelty_score
            + self.w_intent_stability * signals.intent_stability
            + self.w_completeness * signals.completeness
            + self.w_evidence_gap * signals.evidence_gap
        )
        negative = self.w_retrieval_cost * adjusted_cost

        raw_utility = positive - negative
        utility = max(0.0, min(1.0, raw_utility))

        breakdown = {
            "novelty_contribution": self.w_novelty * signals.novelty_score,
            "stability_contribution": self.w_intent_stability * signals.intent_stability,
            "completeness_contribution": self.w_completeness * signals.completeness,
            "evidence_gap_contribution": self.w_evidence_gap * signals.evidence_gap,
            "cost_deduction": negative,
            "time_penalty": time_penalty,
        }

        return UtilityResult(utility=utility, signals=signals, breakdown=breakdown)


# ──────────────────────────────────────────────────────────────
# Utility Estimator (facade)
# ──────────────────────────────────────────────────────────────

class RetrievalUtilityEstimator:
    """
    Facade over a UtilityPolicy.
    Accepts a policy at construction time; defaults to WeightedLinearPolicy.

    Usage:
        estimator = RetrievalUtilityEstimator()
        result = estimator.compute(signals)
    """

    def __init__(self, policy: Optional[UtilityPolicy] = None) -> None:
        self.policy = policy or WeightedLinearPolicy()
        logger.info("Using utility policy: %s", self.policy.name)

    def compute(self, signals: UtilitySignals) -> UtilityResult:
        return self.policy.compute(signals)

    def compute_from_components(
        self,
        novelty_score: float = 0.0,
        intent_stability: float = 0.0,
        completeness: float = 0.0,
        evidence_gap: float = 1.0,
        retrieval_cost: float = 0.1,
        seconds_since_last_retrieval: float = float("inf"),
        chunk_index: int = 0,
    ) -> UtilityResult:
        signals = UtilitySignals(
            novelty_score=novelty_score,
            intent_stability=intent_stability,
            completeness=completeness,
            evidence_gap=evidence_gap,
            retrieval_cost=retrieval_cost,
            seconds_since_last_retrieval=seconds_since_last_retrieval,
            chunk_index=chunk_index,
        )
        return self.compute(signals)
