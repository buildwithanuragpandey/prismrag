"""
PRISM Query Completeness Estimator

Estimates how "complete" a query is, independent of its content.
Completeness is a linguistic/structural signal, not semantic.

Factors considered:
  - Token count relative to typical complete query length
  - Presence of action verbs
  - Presence of specific constraints (numbers, dates, named entities)
  - Sentence structure completeness (ends with verb phrase)
  - Change in length between chunks

This is a heuristic. More sophisticated NLP can be plugged in later.
"""

from __future__ import annotations

import re
from typing import List, Optional


# Typical complete query token count (heuristic upper bound)
COMPLETE_QUERY_TOKEN_TARGET = 25

# Keywords that indicate a query is asking for something specific
ACTION_VERBS = [
    "find", "get", "show", "list", "compare", "explain", "describe",
    "what", "who", "when", "where", "how", "why", "which", "calculate",
    "summarize", "plan", "organize", "book", "cancel", "need", "want",
    "help", "tell", "give", "provide", "suggest", "recommend",
]

CONSTRAINT_PATTERNS = [
    r"\b\d+\b",                    # Any number
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b\d{4}\b",                  # Year
    r"\b(international|domestic|global|local)\b",
    r"\b(before|after|within|during|between)\b",
    r"\b(more than|less than|at least|at most|up to)\b",
    r"\b(cancellation|policy|catering|capacity|budget|cost|price)\b",
    r"'s\b",                       # Possessive → specific entity
]


def count_constraints(text: str) -> int:
    """Count how many constraint patterns match in the text."""
    text_lower = text.lower()
    count = 0
    for pattern in CONSTRAINT_PATTERNS:
        if re.search(pattern, text_lower):
            count += 1
    return count


def has_action_verb(text: str) -> bool:
    text_lower = text.lower()
    return any(f" {v} " in f" {text_lower} " or text_lower.startswith(v) for v in ACTION_VERBS)


def estimate_completeness(
    text: str,
    previous_text: Optional[str] = None,
) -> float:
    """
    Estimate query completeness as a float in [0, 1].

    Higher = more complete.

    The estimate is NOT perfect — it is a heuristic proxy.
    Logged and exposed so researchers can audit it.
    """
    if not text or not text.strip():
        return 0.0

    words = text.strip().split()
    word_count = len(words)

    # Component 1: Length score (sigmoid-like, saturates near target)
    length_score = min(1.0, word_count / COMPLETE_QUERY_TOKEN_TARGET)

    # Component 2: Has action verb
    verb_score = 0.2 if has_action_verb(text) else 0.0

    # Component 3: Number of specific constraints
    constraint_count = count_constraints(text)
    constraint_score = min(0.30, constraint_count * 0.08)

    # Component 4: Ends with a complete thought (ends with period/question mark
    # or the last word is a noun/adjective — very rough)
    punctuation_bonus = 0.05 if text.rstrip().endswith((".", "?", "!")) else 0.0

    # Component 5: Growth bonus (if query grew vs. previous)
    growth_bonus = 0.0
    if previous_text:
        prev_words = len(previous_text.strip().split())
        if word_count > prev_words:
            growth_bonus = min(0.10, (word_count - prev_words) / 20.0)

    # Weighted sum
    raw = (
        length_score * 0.45
        + verb_score
        + constraint_score
        + punctuation_bonus
        + growth_bonus
    )

    return round(min(1.0, raw), 4)
