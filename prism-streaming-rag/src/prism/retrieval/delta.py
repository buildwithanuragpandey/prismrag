"""
PRISM Delta Retriever (Phase 14)

Implements targeted delta retrieval for late-constraint refinement.

When a user adds new information AFTER an answer has been given:
  old_query + new_constraint = delta_query

The DeltaRetriever:
  1. Builds a delta_query from the new constraint
  2. Retrieves against only the delta_query
  3. Deduplicates against existing evidence
  4. Measures delta_retrieval_cost vs full_retrieval_cost

This is a key research metric: REFINEMENT EFFICIENCY
  = delta_retrieval_cost / full_retrieval_cost
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .hybrid import PRISMHybridRetriever, RetrievedChunk, RetrievalResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Delta Query Builder
# ──────────────────────────────────────────────────────────────

def build_delta_query(
    original_query: str,
    new_constraint: str,
) -> str:
    """
    Build a focused delta query from the original query + new constraint.
    The delta query targets only the new information, not the full query.

    This is a simple heuristic baseline. More sophisticated approaches
    (e.g., constraint extraction + targeted expansion) can be plugged in.
    """
    if not original_query:
        return new_constraint
    if not new_constraint:
        return original_query

    # Focus on the new constraint + key terms from original
    return f"{new_constraint} {original_query}"


# ──────────────────────────────────────────────────────────────
# Delta Retrieval Result
# ──────────────────────────────────────────────────────────────

@dataclass
class DeltaRetrievalResult:
    """Result of a delta retrieval operation."""

    delta_retrieval_id: str
    original_query: str
    new_constraint: str
    delta_query: str
    new_chunks: List[RetrievedChunk]         # Chunks not in existing evidence
    duplicate_chunk_ids: List[str]           # Chunks already in evidence
    delta_latency_ms: float
    full_retrieval_estimate_ms: float = 0.0   # For efficiency comparison

    @property
    def refinement_efficiency(self) -> float:
        """delta_cost / full_cost. Lower = more efficient delta retrieval."""
        if self.full_retrieval_estimate_ms == 0:
            return 1.0
        return self.delta_latency_ms / self.full_retrieval_estimate_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_retrieval_id": self.delta_retrieval_id,
            "original_query": self.original_query,
            "new_constraint": self.new_constraint,
            "delta_query": self.delta_query,
            "new_chunk_count": len(self.new_chunks),
            "duplicate_count": len(self.duplicate_chunk_ids),
            "delta_latency_ms": round(self.delta_latency_ms, 1),
            "full_retrieval_estimate_ms": round(self.full_retrieval_estimate_ms, 1),
            "refinement_efficiency": round(self.refinement_efficiency, 4),
            "new_chunks": [c.to_dict() for c in self.new_chunks],
        }


# ──────────────────────────────────────────────────────────────
# Delta Retriever
# ──────────────────────────────────────────────────────────────

class DeltaRetriever:
    """
    Performs targeted delta retrieval for late-constraint refinement.

    Instead of re-running the full RAG pipeline when new constraints arrive,
    the DeltaRetriever retrieves only for the NEW information.

    Args:
        retriever: PRISMHybridRetriever instance
    """

    def __init__(self, retriever: PRISMHybridRetriever) -> None:
        self.retriever = retriever

    def retrieve_delta(
        self,
        original_query: str,
        new_constraint: str,
        existing_chunk_ids: Set[str],
        full_retrieval_estimate_ms: float = 0.0,
    ) -> DeltaRetrievalResult:
        """
        Retrieve new evidence for a new constraint.

        Args:
            original_query: The original user query
            new_constraint: The new constraint or additional detail
            existing_chunk_ids: Set of chunk IDs already in evidence store
                                 (used to deduplicate)
            full_retrieval_estimate_ms: For efficiency measurement

        Returns:
            DeltaRetrievalResult with new (non-duplicate) chunks
        """
        t_start = time.time()
        delta_id = str(uuid.uuid4())

        delta_query = build_delta_query(original_query, new_constraint)

        logger.info(
            "Delta retrieval: constraint='%s' delta_query='%s'",
            new_constraint,
            delta_query,
        )

        result = self.retriever.retrieve(delta_query)
        all_chunks = result.chunks

        # Deduplicate against existing evidence
        new_chunks = [c for c in all_chunks if c.chunk_id not in existing_chunk_ids]
        dup_ids = [c.chunk_id for c in all_chunks if c.chunk_id in existing_chunk_ids]

        delta_latency_ms = (time.time() - t_start) * 1000

        logger.info(
            "Delta retrieved %d new chunks (%d duplicates filtered) in %.1fms",
            len(new_chunks),
            len(dup_ids),
            delta_latency_ms,
        )

        return DeltaRetrievalResult(
            delta_retrieval_id=delta_id,
            original_query=original_query,
            new_constraint=new_constraint,
            delta_query=delta_query,
            new_chunks=new_chunks,
            duplicate_chunk_ids=dup_ids,
            delta_latency_ms=delta_latency_ms,
            full_retrieval_estimate_ms=full_retrieval_estimate_ms,
        )

    async def retrieve_delta_async(
        self,
        original_query: str,
        new_constraint: str,
        existing_chunk_ids: Set[str],
        full_retrieval_estimate_ms: float = 0.0,
    ) -> DeltaRetrievalResult:
        """Async version of retrieve_delta."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.retrieve_delta,
            original_query,
            new_constraint,
            existing_chunk_ids,
            full_retrieval_estimate_ms,
        )
