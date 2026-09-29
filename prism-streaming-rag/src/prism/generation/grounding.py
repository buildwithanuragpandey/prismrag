"""
PRISM Citation Validator and Grounding Checker (Phase 15)

CitationValidator:
  - Verifies that every inline citation [DOC-ID, CHUNK-ID] in an answer
    maps to an actual EvidenceItem in the store
  - Returns list of invalid citations

GroundingChecker:
  - For each answer claim, checks if it is supported by evidence
  - Uses token overlap heuristic (extensible to NLI later)

UnsupportedClaimDetector:
  - Flags claims that have no grounding support
  - Returns structured report for telemetry + frontend display
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ──────────────────────────────────────────────────────────────
# Citation Validator
# ──────────────────────────────────────────────────────────────

CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")


@dataclass
class CitationValidationResult:
    valid: bool
    valid_citations: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    citation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "citation_count": self.citation_count,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
        }


class CitationValidator:
    """
    Validates that all inline citations in an answer text
    refer to actual evidence in the store.

    Citation format expected: [doc_id, chunk_id] or [evidence_id]
    """

    def validate(
        self,
        answer_text: str,
        evidence_store,          # EvidenceStore
    ) -> CitationValidationResult:
        """
        Check all citations in answer_text against the evidence store.
        """
        cited = CITATION_PATTERN.findall(answer_text)
        if not cited:
            # No citations at all — this is technically a grounding concern
            # but not a citation format error
            return CitationValidationResult(
                valid=True,
                citation_count=0,
                valid_citations=[],
                invalid_citations=[],
            )

        all_items = evidence_store.get_all(include_stale=True)
        valid_chunk_ids: Set[str] = {item.chunk_id for item in all_items}
        valid_doc_ids: Set[str] = {item.document_id for item in all_items}
        valid_ev_ids: Set[str] = {item.evidence_id for item in all_items}

        valid = []
        invalid = []

        for citation_text in cited:
            parts = [p.strip() for p in citation_text.split(",")]
            # Check if any part matches a known ID
            is_valid = any(
                p in valid_chunk_ids or p in valid_doc_ids or p in valid_ev_ids
                for p in parts
            )
            if is_valid:
                valid.append(f"[{citation_text}]")
            else:
                invalid.append(f"[{citation_text}]")

        return CitationValidationResult(
            valid=len(invalid) == 0,
            valid_citations=valid,
            invalid_citations=invalid,
            citation_count=len(cited),
        )


# ──────────────────────────────────────────────────────────────
# Grounding Checker
# ──────────────────────────────────────────────────────────────

@dataclass
class ClaimGroundingResult:
    claim_text: str
    is_supported: bool
    supporting_evidence_ids: List[str] = field(default_factory=list)
    overlap_score: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text[:200],
            "is_supported": self.is_supported,
            "supporting_evidence_ids": self.supporting_evidence_ids,
            "overlap_score": round(self.overlap_score, 4),
            "note": self.note,
        }


class GroundingChecker:
    """
    Checks whether answer claims are grounded in retrieved evidence.

    Method: token overlap heuristic (baseline).
    Can be upgraded to NLI-based approach (e.g., DeBERTa) later.

    Minimum overlap threshold is configurable.
    """

    def __init__(self, min_overlap: float = 0.15) -> None:
        self.min_overlap = min_overlap

    def _token_overlap(self, claim: str, evidence_text: str) -> float:
        """Compute token overlap between claim and evidence text."""
        claim_tokens = set(
            t for t in claim.lower().split()
            if len(t) > 3
        )
        ev_tokens = set(evidence_text.lower().split())
        if not claim_tokens:
            return 0.0
        intersection = claim_tokens & ev_tokens
        return len(intersection) / len(claim_tokens)

    def check_claim(
        self,
        claim_text: str,
        evidence_store,
    ) -> ClaimGroundingResult:
        """Check if a single claim is supported by any evidence item."""
        active = evidence_store.get_all()
        if not active:
            return ClaimGroundingResult(
                claim_text=claim_text,
                is_supported=False,
                note="no_evidence_in_store",
            )

        best_score = 0.0
        best_ev_ids = []

        for item in active:
            score = self._token_overlap(claim_text, item.text)
            if score > best_score:
                best_score = score
                best_ev_ids = [item.evidence_id]
            elif score == best_score and score > 0:
                best_ev_ids.append(item.evidence_id)

        is_supported = best_score >= self.min_overlap
        return ClaimGroundingResult(
            claim_text=claim_text,
            is_supported=is_supported,
            supporting_evidence_ids=best_ev_ids if is_supported else [],
            overlap_score=best_score,
            note="" if is_supported else "insufficient_overlap",
        )

    def check_all_claims(
        self,
        claims,      # List[Claim]
        evidence_store,
    ) -> List[ClaimGroundingResult]:
        """Check grounding for all claims."""
        return [self.check_claim(c.text, evidence_store) for c in claims]


# ──────────────────────────────────────────────────────────────
# Unsupported Claim Detector
# ──────────────────────────────────────────────────────────────

@dataclass
class GroundingReport:
    answer_version: int
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    grounding_ratio: float
    unsupported_claim_texts: List[str] = field(default_factory=list)
    citation_result: Optional[CitationValidationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_version": self.answer_version,
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "grounding_ratio": round(self.grounding_ratio, 4),
            "unsupported_claim_texts": self.unsupported_claim_texts,
            "citation_validation": self.citation_result.to_dict() if self.citation_result else None,
        }


class UnsupportedClaimDetector:
    """
    Detects unsupported claims in an answer version.
    Combines GroundingChecker + CitationValidator into one report.
    """

    def __init__(
        self,
        grounding_checker: Optional[GroundingChecker] = None,
        citation_validator: Optional[CitationValidator] = None,
    ) -> None:
        self.grounding_checker = grounding_checker or GroundingChecker()
        self.citation_validator = citation_validator or CitationValidator()

    def analyze(self, answer_version, evidence_store) -> GroundingReport:
        """
        Full grounding analysis of an AnswerVersion.
        """
        claims = answer_version.claims
        grounding_results = self.grounding_checker.check_all_claims(claims, evidence_store)
        citation_result = self.citation_validator.validate(answer_version.text, evidence_store)

        supported = [r for r in grounding_results if r.is_supported]
        unsupported = [r for r in grounding_results if not r.is_supported]
        ratio = len(supported) / len(claims) if claims else 0.0

        return GroundingReport(
            answer_version=answer_version.version,
            total_claims=len(claims),
            supported_claims=len(supported),
            unsupported_claims=len(unsupported),
            grounding_ratio=ratio,
            unsupported_claim_texts=[r.claim_text for r in unsupported],
            citation_result=citation_result,
        )
