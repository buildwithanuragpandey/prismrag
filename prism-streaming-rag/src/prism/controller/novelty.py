"""
PRISM Semantic Novelty Detector (Phase 5)

Computes how "novel" the current query state is compared to the previous state.
Novel = new information has arrived that the retrieval system hasn't seen.

Method:
  - Embed current_text and previous_text using the embedding model
  - Compute cosine distance (1 - cosine_similarity)
  - Normalize to [0, 1]

High novelty → system should consider retrieval.
Low novelty → query is still evolving in the same direction or unchanged.

NOTE: This uses the configured embedding model (same as retrieval).
Weights are configurable. Results are transparent (logged to telemetry).
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 1.0  # identical if both zero
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


class SemanticNoveltyDetector:
    """
    Computes novelty score for each new transcript chunk.

    The novelty score measures how much new semantic content has arrived
    since the previous retrieval or transcript snapshot.

    Score = cosine_distance(embed(current), embed(previous))
           = 1 - cosine_similarity(...)

    Edge cases:
      - First chunk (no previous): novelty = 1.0 (always novel)
      - Identical text: novelty = 0.0
      - Empty delta: novelty = 0.0

    Args:
        embedding_fn: Callable that takes a string and returns a float vector.
                      Injected so the detector is model-agnostic.
        high_threshold: Above this → high novelty
        low_threshold: Below this → low novelty (suppress candidate)
    """

    def __init__(
        self,
        embedding_fn,
        high_threshold: float = 0.30,
        low_threshold: float = 0.05,
    ) -> None:
        self._embed = embedding_fn
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._cache: dict[str, List[float]] = {}

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding with simple cache."""
        if text not in self._cache:
            self._cache[text] = self._embed(text)
        return self._cache[text]

    def compute(
        self,
        current_text: str,
        previous_text: str,
    ) -> float:
        """
        Compute novelty score between current and previous transcript.

        Returns:
            novelty_score in [0, 1]
              0.0 = no change
              1.0 = completely different
        """
        if not current_text:
            return 0.0

        if not previous_text:
            # First chunk → maximally novel
            return 1.0

        if current_text == previous_text:
            return 0.0

        try:
            emb_current = self._get_embedding(current_text)
            emb_previous = self._get_embedding(previous_text)
            sim = cosine_similarity(emb_current, emb_previous)
            # Cosine distance = 1 - similarity, clipped to [0, 1]
            novelty = max(0.0, min(1.0, 1.0 - sim))
            return novelty
        except Exception as exc:
            logger.warning("Novelty computation failed: %s. Returning 0.5.", exc)
            return 0.5

    def is_high_novelty(self, score: float) -> bool:
        return score >= self.high_threshold

    def is_low_novelty(self, score: float) -> bool:
        return score <= self.low_threshold

    def delta_novelty(
        self,
        current_text: str,
        previous_retrieval_text: str,
    ) -> float:
        """
        Compute novelty relative to the LAST RETRIEVAL point.
        Used to detect if enough has changed since we last retrieved
        (regardless of previous chunk).
        """
        return self.compute(current_text, previous_retrieval_text)
