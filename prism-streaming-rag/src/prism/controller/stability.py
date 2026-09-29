"""
PRISM Intent Stability Tracker (Phase 6)

Tracks intent predictions over multiple transcript chunks and computes
a stability score.

Stability = how consistently the system is detecting the same intent(s).

High stability → the user's intent is clear and unlikely to change
                 → good time to retrieve
Low stability  → intent is still evolving / unknown
                 → better to WAIT

The stability window is configurable.

Implementation notes:
  - We track the set of detected intent labels per chunk.
  - Stability is computed using Jaccard similarity of consecutive intent sets.
  - Over a sliding window, we compute average pairwise similarity.
  - This is a transparent, deterministic algorithm — not a learned model.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    intersection = set_a & set_b
    if not union:
        return 1.0
    return len(intersection) / len(union)


class IntentStabilityTracker:
    """
    Tracks intent stability across transcript chunks.

    At each chunk, the detected intents are stored.
    Stability is computed as average Jaccard similarity across the
    configured sliding window.

    Args:
        stable_window: Number of consecutive chunks to compare. Default 2.
                       If intent remains the same for `stable_window` chunks,
                       stability approaches 1.0.
        max_intents: Maximum intents to track per chunk (caps complexity).
    """

    def __init__(
        self,
        stable_window: int = 2,
        max_intents: int = 5,
    ) -> None:
        self.stable_window = stable_window
        self.max_intents = max_intents
        # Per-session history: session_id → deque of intent label sets
        self._history: Dict[str, deque] = {}

    def _get_history(self, session_id: str) -> deque:
        if session_id not in self._history:
            self._history[session_id] = deque(maxlen=max(self.stable_window + 1, 5))
        return self._history[session_id]

    def update(self, session_id: str, intent_labels: List[str]) -> float:
        """
        Record intent labels for this chunk and return stability score.

        Returns:
            stability_score in [0, 1]
              0.0 = no stable pattern detected
              1.0 = perfect intent consistency across window
        """
        history = self._get_history(session_id)

        # Cap at max_intents to avoid complexity explosion
        label_set = set(intent_labels[: self.max_intents])
        history.append(label_set)

        return self.compute_stability(session_id)

    def compute_stability(self, session_id: str) -> float:
        """
        Compute stability from history.

        Returns 0.0 if we have fewer chunks than the stable_window.
        """
        history = self._get_history(session_id)

        if len(history) < 2:
            return 0.0  # Not enough data yet

        # Compute pairwise Jaccard for the last `stable_window` pairs
        window = list(history)[-self.stable_window - 1:]
        similarities = []
        for i in range(len(window) - 1):
            sim = jaccard_similarity(window[i], window[i + 1])
            similarities.append(sim)

        if not similarities:
            return 0.0

        stability = sum(similarities) / len(similarities)
        return round(stability, 4)

    def is_stable(self, session_id: str, threshold: float = 0.5) -> bool:
        return self.compute_stability(session_id) >= threshold

    def get_history_labels(self, session_id: str) -> List[List[str]]:
        """Return per-chunk intent label lists for inspection."""
        history = self._get_history(session_id)
        return [sorted(s) for s in history]

    def reset_session(self, session_id: str) -> None:
        if session_id in self._history:
            del self._history[session_id]
