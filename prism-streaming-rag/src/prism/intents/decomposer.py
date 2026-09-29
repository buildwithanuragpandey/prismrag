"""
PRISM Multi-Intent Decomposer (Phase 10, 11)

Detects and decomposes multiple intents from a single query.
Runs parallel sub-query retrieval for each detected intent.

Example:
  Input:  "I need a venue for 30 people, cancellation policy and catering."
  Output: [
    DetectedIntent(label="venue_selection", sub_query="venue capacity 30 people"),
    DetectedIntent(label="cancellation_policy", sub_query="cancellation policy"),
    DetectedIntent(label="catering", sub_query="catering options"),
  ]

Implementation:
  - Uses LLM for decomposition (configurable, model-agnostic interface)
  - Falls back to keyword/pattern heuristic if LLM unavailable
  - Results are tracked per-session for stability computation
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from prism.controller.query_state import DetectedIntent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Heuristic Decomposer (no LLM required)
# ──────────────────────────────────────────────────────────────

INTENT_PATTERNS = [
    (r"\b(venue|location|place|room|hall|hotel|conference)\b", "venue_selection"),
    (r"\b(cancellation|cancel)\b", "cancellation_policy"),
    (r"\b(catering|food|meal|beverage|lunch|dinner|breakfast)\b", "catering"),
    (r"\b(budget|cost|price|fee|expense|reimburs)\b", "budget_cost"),
    (r"\b(capacity|people|person|attendee|participant|seat)\b", "capacity"),
    (r"\b(travel|transport|flight|hotel|accommodation|book)\b", "travel_logistics"),
    (r"\b(policy|regulation|rule|requirement|guideline)\b", "policy"),
    (r"\b(international|overseas|abroad|global)\b", "international_context"),
    (r"\b(schedule|time|date|agenda|session|slot)\b", "scheduling"),
    (r"\b(speaker|presenter|trainer|facilitator)\b", "speakers"),
    (r"\b(registration|sign.?up|enroll)\b", "registration"),
    (r"\b(refund|reimburs)\b", "reimbursement"),
    (r"\b(legal|contract|liability|insurance)\b", "legal"),
    (r"\b(tech|equipment|audio|video|projector|screen)\b", "av_tech"),
]


def extract_intents_heuristic(text: str) -> List[DetectedIntent]:
    """
    Pattern-based multi-intent detection (no LLM required).
    Used as fallback and for fast unit testing.
    """
    text_lower = text.lower()
    found: List[DetectedIntent] = []
    seen_labels = set()

    for pattern, label in INTENT_PATTERNS:
        if re.search(pattern, text_lower) and label not in seen_labels:
            seen_labels.add(label)
            # Build a sub-query from the matched context
            match = re.search(pattern, text_lower)
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 50)
            snippet = text[start:end].strip()

            found.append(
                DetectedIntent(
                    label=label,
                    description=label.replace("_", " ").title(),
                    confidence=0.7,
                    sub_query=snippet,
                )
            )

    # If nothing found, treat entire text as one intent
    if not found:
        found.append(
            DetectedIntent(
                label="general_query",
                description="General information query",
                confidence=0.5,
                sub_query=text,
            )
        )

    return found


# ──────────────────────────────────────────────────────────────
# LLM-Based Decomposer
# ──────────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM_PROMPT = """You are a precise query analyzer for a retrieval system.
Given a user query, identify all distinct information needs (intents).
For each intent, provide:
- label: snake_case identifier
- description: one line description
- sub_query: the best retrieval query to satisfy this intent

Respond ONLY as a JSON array. Example:
[
  {"label": "venue_capacity", "description": "Find venues for 30 people", "sub_query": "venue capacity 30 people meeting room"},
  {"label": "catering", "description": "Catering options", "sub_query": "catering options corporate event"}
]

Keep sub_queries concise and retrieval-optimized. Maximum 5 intents."""


def extract_intents_llm(
    text: str,
    llm_fn: Callable[[str, str], str],
    max_intents: int = 5,
) -> List[DetectedIntent]:
    """
    LLM-based multi-intent extraction.

    Args:
        text: Query text
        llm_fn: Callable(system_prompt, user_prompt) → response string
        max_intents: Cap on returned intents
    """
    import json

    try:
        response = llm_fn(DECOMPOSE_SYSTEM_PROMPT, text)
        # Extract JSON array from response
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if not match:
            logger.warning("No JSON array in LLM response, falling back to heuristic")
            return extract_intents_heuristic(text)

        raw = json.loads(match.group())
        intents = []
        for item in raw[:max_intents]:
            intents.append(
                DetectedIntent(
                    label=item.get("label", "unknown"),
                    description=item.get("description", ""),
                    confidence=0.85,
                    sub_query=item.get("sub_query", text),
                )
            )
        return intents if intents else extract_intents_heuristic(text)

    except Exception as exc:
        logger.warning("LLM intent extraction failed: %s. Using heuristic.", exc)
        return extract_intents_heuristic(text)


# ──────────────────────────────────────────────────────────────
# Multi-Intent Decomposer
# ──────────────────────────────────────────────────────────────

class MultiIntentDecomposer:
    """
    Detects multiple intents from a query and decomposes into sub-queries.

    Supports two backends:
      - "heuristic": Fast, no LLM (default)
      - "llm": Uses configured LLM for better quality

    Sub-queries are retrieved in parallel via ParallelSubQueryRetriever.

    Args:
        backend: "heuristic" | "llm"
        llm_fn: Required if backend="llm". Callable(sys_prompt, user_prompt) → str
        max_intents: Maximum intents to return
    """

    def __init__(
        self,
        backend: str = "heuristic",
        llm_fn: Optional[Callable[[str, str], str]] = None,
        max_intents: int = 5,
    ) -> None:
        self.backend = backend
        self.llm_fn = llm_fn
        self.max_intents = max_intents

        if backend == "llm" and llm_fn is None:
            logger.warning(
                "backend='llm' but no llm_fn provided — falling back to heuristic"
            )
            self.backend = "heuristic"

    def decompose(self, text: str) -> List[DetectedIntent]:
        """
        Decompose a query into intents.

        Returns list of DetectedIntent objects, each with a sub_query.
        """
        if not text or not text.strip():
            return []

        if self.backend == "llm" and self.llm_fn is not None:
            intents = extract_intents_llm(text, self.llm_fn, self.max_intents)
        else:
            intents = extract_intents_heuristic(text)

        return intents[: self.max_intents]

    def get_sub_queries(self, text: str) -> List[str]:
        """Convenience method: decompose and return just the sub-query strings."""
        intents = self.decompose(text)
        return [i.sub_query for i in intents if i.sub_query]
