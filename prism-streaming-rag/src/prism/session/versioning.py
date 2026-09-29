"""
PRISM Answer Version Manager (Phase 12) and
Late Constraint Detector (Phase 13)

Answer Versioning:
  - Each answer (V1, V2, V3...) is stored with full provenance
  - Claims are tracked per version
  - Diffs between versions show what changed and why

Late Constraint Detector:
  - Detects when new user input contains constraints that invalidate
    or affect previously generated claims
  - Does NOT restart the full pipeline
  - Instead triggers DeltaRetriever + AnswerRefiner

AnswerRefiner:
  - Generates V2 from preserved evidence + new delta evidence
  - Shows exactly which claims changed
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ──────────────────────────────────────────────────────────────
# Claim
# ──────────────────────────────────────────────────────────────

@dataclass
class Claim:
    """An atomic factual claim in an answer."""

    claim_id: str
    text: str
    evidence_ids: List[str] = field(default_factory=list)  # Supporting evidence
    is_grounded: bool = False
    grounding_score: float = 0.0
    is_affected_by_constraint: bool = False
    is_supported: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "evidence_ids": self.evidence_ids,
            "is_grounded": self.is_grounded,
            "grounding_score": round(self.grounding_score, 4),
            "is_affected_by_constraint": self.is_affected_by_constraint,
            "is_supported": self.is_supported,
        }


# ──────────────────────────────────────────────────────────────
# Answer Version
# ──────────────────────────────────────────────────────────────

@dataclass
class AnswerVersion:
    """
    A single versioned answer with full provenance.

    Stores:
      - The answer text
      - All claims with grounding
      - Evidence IDs used
      - What triggered this version
      - What changed vs. previous version
    """

    version: int
    text: str
    session_id: str
    answer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Claims
    claims: List[Claim] = field(default_factory=list)

    # Evidence
    evidence_ids: List[str] = field(default_factory=list)
    context_used: str = ""

    # Trigger
    trigger: str = "initial"   # "initial" | "new_constraint" | "refinement"
    trigger_constraint: str = ""

    # Diff vs previous version
    changed_claims: List[str] = field(default_factory=list)    # claim texts
    unchanged_claims: List[str] = field(default_factory=list)  # claim texts
    new_evidence_ids: List[str] = field(default_factory=list)

    # Metrics
    grounding_score: float = 0.0       # Average claim grounding
    citation_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "answer_id": self.answer_id,
            "text": self.text,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "trigger": self.trigger,
            "trigger_constraint": self.trigger_constraint,
            "claim_count": len(self.claims),
            "claims": [c.to_dict() for c in self.claims],
            "evidence_ids": self.evidence_ids,
            "changed_claims": self.changed_claims,
            "unchanged_claims": self.unchanged_claims,
            "new_evidence_ids": self.new_evidence_ids,
            "grounding_score": round(self.grounding_score, 4),
            "citation_valid": self.citation_valid,
        }


# ──────────────────────────────────────────────────────────────
# Answer Version Manager
# ──────────────────────────────────────────────────────────────

class AnswerVersionManager:
    """
    Manages answer versions for a session.

    On first answer: creates V1
    On refinement:   creates V2, V3...

    Computes claim-level diffs between versions.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._versions: List[AnswerVersion] = []

    def add_version(
        self,
        text: str,
        claims: List[Claim],
        evidence_ids: List[str],
        context_used: str = "",
        trigger: str = "initial",
        trigger_constraint: str = "",
        new_evidence_ids: Optional[List[str]] = None,
    ) -> AnswerVersion:
        """Create and store a new answer version."""
        version_num = len(self._versions) + 1
        prev = self._versions[-1] if self._versions else None

        # Compute claim diff
        changed, unchanged = self._compute_claim_diff(
            prev.claims if prev else [], claims
        )

        # Average grounding
        grounded = [c for c in claims if c.is_grounded]
        grounding = len(grounded) / len(claims) if claims else 0.0

        v = AnswerVersion(
            version=version_num,
            text=text,
            session_id=self.session_id,
            claims=claims,
            evidence_ids=evidence_ids,
            context_used=context_used,
            trigger=trigger,
            trigger_constraint=trigger_constraint,
            changed_claims=changed,
            unchanged_claims=unchanged,
            new_evidence_ids=new_evidence_ids or [],
            grounding_score=grounding,
        )

        self._versions.append(v)
        return v

    def _compute_claim_diff(
        self, prev_claims: List[Claim], curr_claims: List[Claim]
    ) -> Tuple[List[str], List[str]]:
        """Simple text-based claim diff."""
        prev_texts = {c.text.strip() for c in prev_claims}
        curr_texts = {c.text.strip() for c in curr_claims}

        changed = [t for t in curr_texts if t not in prev_texts]
        unchanged = [t for t in curr_texts if t in prev_texts]
        return changed, unchanged

    @property
    def latest(self) -> Optional[AnswerVersion]:
        return self._versions[-1] if self._versions else None

    @property
    def version_count(self) -> int:
        return len(self._versions)

    def get_all_versions(self) -> List[AnswerVersion]:
        return list(self._versions)

    def get_version(self, version: int) -> Optional[AnswerVersion]:
        for v in self._versions:
            if v.version == version:
                return v
        return None


# ──────────────────────────────────────────────────────────────
# Late Constraint Detector (Phase 13)
# ──────────────────────────────────────────────────────────────

# Keywords that signal a new constraint is being added
CONSTRAINT_SIGNALS = [
    r"\b(actually|wait|also|but|however|additionally|and)\b",
    r"\b(international|overseas|abroad|domestic)\b",
    r"\b(after|before|during|within|between)\b",
    r"\b(cancel|cancelled|changed|updated|revised)\b",
    r"\b(more than|less than|at least|at most|exactly)\b",
    r"\b\d+\b",
    r"\b(the trip|the event|the booking|the order|the policy)\b",
    r"\b(it was|it is|they were|he was|she was)\b",
]

# Patterns that indicate constraint type
CONSTRAINT_TYPES = {
    "temporal": [r"\b(before|after|during|within|since|until|when)\b"],
    "geographic": [r"\b(international|domestic|overseas|abroad|in \w+)\b"],
    "quantitative": [r"\b\d+\b", r"\b(more than|less than|at least)\b"],
    "conditional": [r"\b(if|only if|when|provided that|assuming)\b"],
    "negation": [r"\b(not|no|without|except|unless)\b"],
    "refinement": [r"\b(specifically|particularly|especially|in particular)\b"],
}


@dataclass
class DetectedConstraint:
    text: str
    constraint_type: str
    confidence: float
    affected_intent_labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "constraint_type": self.constraint_type,
            "confidence": round(self.confidence, 4),
            "affected_intent_labels": self.affected_intent_labels,
        }


class LateConstraintDetector:
    """
    Detects new constraints in post-answer user input.

    When the user adds information after an answer has been given,
    this component determines:
      1. Is this a new constraint? (not just reformulation)
      2. What type of constraint is it?
      3. Which answer claims does it potentially affect?

    Triggers delta retrieval + answer refinement only when needed.

    Implementation: heuristic rule-based baseline.
    A learned classifier can be swapped in via the same interface.
    """

    def __init__(self) -> None:
        pass

    def detect(
        self,
        new_text: str,
        previous_text: str,
        previous_claims: List[Claim],
    ) -> List[DetectedConstraint]:
        """
        Detect constraints in new_text that weren't in previous_text.

        Returns list of DetectedConstraint objects.
        Returns empty list if no meaningful new constraints found.
        """
        # Extract delta (what's new)
        delta = self._extract_delta(new_text, previous_text)
        if not delta.strip():
            return []

        constraints = []
        delta_lower = delta.lower()

        # Check each signal pattern
        has_signal = any(
            re.search(pat, delta_lower) for pat in CONSTRAINT_SIGNALS
        )

        if not has_signal:
            return []

        # Detect constraint type(s)
        for c_type, patterns in CONSTRAINT_TYPES.items():
            for pat in patterns:
                if re.search(pat, delta_lower):
                    # Find potentially affected claims
                    affected = self._find_affected_claims(delta, previous_claims)

                    constraints.append(DetectedConstraint(
                        text=delta.strip(),
                        constraint_type=c_type,
                        confidence=0.70,
                        affected_intent_labels=affected,
                    ))
                    break  # One constraint per type

        return constraints

    def _extract_delta(self, new_text: str, previous_text: str) -> str:
        """Extract the truly new portion of text."""
        if not previous_text:
            return new_text
        # Simple: return anything after the common prefix
        common_len = 0
        for i, (a, b) in enumerate(zip(new_text, previous_text)):
            if a == b:
                common_len = i + 1
            else:
                break
        return new_text[common_len:].strip()

    def _find_affected_claims(
        self, constraint_text: str, claims: List[Claim]
    ) -> List[str]:
        """
        Heuristic: find which claims share keywords with the new constraint.
        Returns list of claim texts (truncated) that may be affected.
        """
        constraint_tokens = set(
            t for t in constraint_text.lower().split()
            if len(t) > 3
        )
        affected = []
        for claim in claims:
            claim_tokens = set(claim.text.lower().split())
            overlap = constraint_tokens & claim_tokens
            if len(overlap) >= 1:
                affected.append(claim.text[:80])
        return affected


# ──────────────────────────────────────────────────────────────
# Claim Impact Analyzer
# ──────────────────────────────────────────────────────────────

class ClaimImpactAnalyzer:
    """
    Determines which specific claims in the current answer
    are invalidated or need revision given a new constraint.

    Returns:
      - affected_claims: need to be re-generated with new evidence
      - unaffected_claims: can be preserved unchanged
    """

    def analyze(
        self,
        constraint: DetectedConstraint,
        claims: List[Claim],
    ) -> Tuple[List[Claim], List[Claim]]:
        """
        Partition claims into affected and unaffected.

        Args:
            constraint: The detected constraint
            claims: Current answer claims

        Returns:
            (affected_claims, unaffected_claims)
        """
        constraint_tokens = set(
            t for t in constraint.text.lower().split()
            if len(t) > 3
        )

        affected = []
        unaffected = []

        for claim in claims:
            claim_tokens = set(claim.text.lower().split())
            overlap = len(constraint_tokens & claim_tokens)

            if overlap >= 1:
                claim_copy = Claim(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    evidence_ids=claim.evidence_ids,
                    is_grounded=claim.is_grounded,
                    grounding_score=claim.grounding_score,
                    is_affected_by_constraint=True,
                )
                affected.append(claim_copy)
            else:
                unaffected.append(claim)

        return affected, unaffected
