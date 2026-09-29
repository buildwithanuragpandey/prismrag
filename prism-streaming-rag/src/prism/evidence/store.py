"""
PRISM Evidence State Manager (Phase 11)

Maintains a session-persistent evidence store.

Each piece of evidence (a retrieved chunk selected for the answer) is
tracked with full provenance:
  - Evidence ID
  - Document ID, Chunk ID
  - Source (URL, title)
  - Retrieval timestamp and retrieval ID
  - The query that produced it
  - Relevance and reranker scores
  - Associated intent
  - Associated claims (populated after generation)

Evidence persists within a session and is NOT discarded between answer versions.
The Late-Constraint Detector uses the evidence store to identify affected claims.

Coverage estimation:
  - Based on query-evidence similarity (heuristic)
  - Used by the controller to suppress unnecessary retrieval
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ──────────────────────────────────────────────────────────────
# Evidence Item
# ──────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    """A single evidence item in the evidence store."""

    evidence_id: str
    chunk_id: str
    document_id: str
    text: str
    source: str = ""
    title: str = ""
    url: str = ""

    # Provenance
    retrieval_id: str = ""
    retrieval_timestamp: float = field(default_factory=time.time)
    chunk_index_at_retrieval: int = 0    # Which transcript chunk triggered this
    query: str = ""                       # Query that produced this evidence

    # Scores
    score: float = 0.0
    rerank_score: Optional[float] = None

    # Linking
    intent_label: str = ""
    claims: List[str] = field(default_factory=list)    # Claims this supports
    is_used_in_answer: bool = False
    answer_versions: List[int] = field(default_factory=list)  # Which answer versions use this

    # Staleness
    is_stale: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text_snippet": self.text[:300],
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "retrieval_id": self.retrieval_id,
            "retrieval_timestamp": self.retrieval_timestamp,
            "chunk_index_at_retrieval": self.chunk_index_at_retrieval,
            "query": self.query,
            "score": round(self.score, 4),
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score else None,
            "intent_label": self.intent_label,
            "claims": self.claims,
            "is_used_in_answer": self.is_used_in_answer,
            "answer_versions": self.answer_versions,
            "is_stale": self.is_stale,
        }

    def citation_key(self) -> str:
        """Short citation reference e.g. [DOC-123, CHUNK-4]."""
        return f"[{self.document_id}, {self.chunk_id}]"


# ──────────────────────────────────────────────────────────────
# Evidence Store
# ──────────────────────────────────────────────────────────────

class EvidenceStore:
    """
    Session-scoped evidence store.

    Tracks all retrieved evidence across the session.
    Evidence is never deleted — it is marked stale when superseded.

    Used by:
      - AnswerVersionManager (to build grounded answers)
      - LateConstraintDetector (to find affected claims)
      - DeltaRetriever (to avoid duplicate retrieval)
      - CitationValidator (to verify inline citations)
      - Coverage Estimator (for controller)
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._items: Dict[str, EvidenceItem] = {}   # evidence_id → item
        self._chunk_ids: Set[str] = set()            # For dedup

    def add_from_retrieval(
        self,
        retrieval_id: str,
        query: str,
        chunks,                    # List[RetrievedChunk]
        chunk_index: int,
        intent_label: str = "",
    ) -> List[EvidenceItem]:
        """
        Add retrieved chunks to the evidence store.
        Returns list of newly added EvidenceItems.
        """
        added = []
        for chunk in chunks:
            # Skip duplicates by chunk_id
            if chunk.chunk_id in self._chunk_ids:
                continue

            ev = EvidenceItem(
                evidence_id=str(uuid.uuid4()),
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                source=chunk.metadata.get("source", ""),
                title=chunk.metadata.get("title", ""),
                url=chunk.metadata.get("url", ""),
                retrieval_id=retrieval_id,
                chunk_index_at_retrieval=chunk_index,
                query=query,
                score=chunk.score,
                rerank_score=chunk.rerank_score,
                intent_label=intent_label,
            )
            self._items[ev.evidence_id] = ev
            self._chunk_ids.add(chunk.chunk_id)
            added.append(ev)

        return added

    def get_all(self, include_stale: bool = False) -> List[EvidenceItem]:
        """Return all evidence items."""
        items = list(self._items.values())
        if not include_stale:
            items = [i for i in items if not i.is_stale]
        return items

    def get_by_intent(self, intent_label: str) -> List[EvidenceItem]:
        return [
            i for i in self._items.values()
            if i.intent_label == intent_label and not i.is_stale
        ]

    def get_chunk_ids(self) -> Set[str]:
        return set(self._chunk_ids)

    def get_evidence_for_answer(self, version: int) -> List[EvidenceItem]:
        return [
            i for i in self._items.values()
            if version in i.answer_versions
        ]

    def mark_used_in_answer(self, evidence_ids: List[str], version: int) -> None:
        for eid in evidence_ids:
            if eid in self._items:
                self._items[eid].is_used_in_answer = True
                if version not in self._items[eid].answer_versions:
                    self._items[eid].answer_versions.append(version)

    def mark_stale(self, evidence_ids: List[str]) -> None:
        for eid in evidence_ids:
            if eid in self._items:
                self._items[eid].is_stale = True

    def estimate_coverage(self, query: str) -> float:
        """
        Heuristic estimate of how well current evidence covers the query.
        Used by the controller for the evidence_gap signal.

        Simple approach: token overlap between query and evidence texts.
        More sophisticated: embedding similarity (added later if needed).
        """
        active_evidence = self.get_all()
        if not active_evidence:
            return 0.0

        query_tokens = set(query.lower().split())
        if not query_tokens:
            return 0.0

        # Check how many query tokens appear in evidence
        evidence_text = " ".join(e.text[:500] for e in active_evidence)
        evidence_tokens = set(evidence_text.lower().split())

        overlap = len(query_tokens & evidence_tokens)
        coverage = overlap / len(query_tokens)

        return min(1.0, round(coverage, 4))

    def to_context_string(self, max_items: int = 10) -> str:
        """Build a formatted context string for LLM generation."""
        active = self.get_all()[:max_items]
        if not active:
            return "No evidence available."

        parts = []
        for ev in active:
            citation = ev.citation_key()
            snippet = ev.text[:600].strip()
            parts.append(f"{citation}\nSource: {ev.title} ({ev.url})\n{snippet}")

        return "\n\n---\n\n".join(parts)

    def summary(self) -> Dict[str, Any]:
        items = list(self._items.values())
        return {
            "session_id": self.session_id,
            "total_items": len(items),
            "active_items": len([i for i in items if not i.is_stale]),
            "stale_items": len([i for i in items if i.is_stale]),
            "unique_documents": len({i.document_id for i in items}),
            "intents_covered": list({i.intent_label for i in items if i.intent_label}),
        }
