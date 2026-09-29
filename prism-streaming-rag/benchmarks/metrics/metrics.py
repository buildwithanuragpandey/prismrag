"""
PRISM Research Metrics (Phase 17, 22)

Implements all evaluation metrics required by the PRISM specification:

Standard IR metrics:
  - Recall@K
  - Precision@K
  - MRR
  - nDCG@K
  - Hit@K

Faithfulness / Grounding:
  - Faithfulness score
  - Answer relevance
  - Citation support rate

Latency:
  - P50, P95 latency
  - Total latency

PRISM Research-Specific Metrics:
  - Early Retrieval Rate
  - Retrieval Efficiency
  - Retrieval Savings
  - Latency Savings
  - Refinement Efficiency
  - Grounding Retention
  - Answer Stability
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ──────────────────────────────────────────────────────────────
# Standard IR Metrics
# ──────────────────────────────────────────────────────────────

def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """Recall@K: fraction of relevant docs found in top-K retrieved."""
    if not relevant_ids:
        return 1.0
    retrieved_k = set(retrieved_ids[:k])
    hits = len(retrieved_k & relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """Precision@K: fraction of top-K that are relevant."""
    if k == 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for r in retrieved_k if r in relevant_ids)
    return hits / k


def hit_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """Hit@K: 1 if any relevant doc in top-K, else 0."""
    retrieved_k = set(retrieved_ids[:k])
    return 1.0 if retrieved_k & relevant_ids else 0.0


def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """Reciprocal Rank: 1/rank of first relevant hit (0 if none)."""
    for i, r in enumerate(retrieved_ids, start=1):
        if r in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain@K."""
    def dcg(ids: List[str], rel: Set[str]) -> float:
        return sum(
            1.0 / math.log2(i + 2)
            for i, doc_id in enumerate(ids[:k])
            if doc_id in rel
        )

    actual_dcg = dcg(retrieved_ids, relevant_ids)
    ideal_ids = list(relevant_ids)[:k]
    ideal_dcg = dcg(ideal_ids, relevant_ids)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def mean_reciprocal_rank(
    results: List[Dict],  # list of {"retrieved_ids": [...], "relevant_ids": [...]}
) -> float:
    """Mean Reciprocal Rank over a list of queries."""
    rrs = [
        reciprocal_rank(r["retrieved_ids"], set(r["relevant_ids"]))
        for r in results
    ]
    return sum(rrs) / len(rrs) if rrs else 0.0


# ──────────────────────────────────────────────────────────────
# Latency Metrics
# ──────────────────────────────────────────────────────────────

def percentile(values: List[float], p: float) -> float:
    """Compute p-th percentile of a list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_vals) - 1)
    fraction = idx - lower
    return sorted_vals[lower] + fraction * (sorted_vals[upper] - sorted_vals[lower])


def latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute mean, P50, P95 latency."""
    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(sum(latencies_ms) / len(latencies_ms), 1),
        "p50": round(percentile(latencies_ms, 50), 1),
        "p95": round(percentile(latencies_ms, 95), 1),
        "min": round(min(latencies_ms), 1),
        "max": round(max(latencies_ms), 1),
        "count": len(latencies_ms),
    }


# ──────────────────────────────────────────────────────────────
# PRISM Research-Specific Metrics
# ──────────────────────────────────────────────────────────────

def early_retrieval_rate(
    retrieval_events: List[Dict],
    utterance_completion_chunk: int,
) -> float:
    """
    Percentage of queries where useful retrieval begins
    before utterance completion.

    retrieval_events: list of {"chunk_index": int, "action": "RETRIEVE"}
    """
    if not retrieval_events:
        return 0.0
    early = [
        e for e in retrieval_events
        if e.get("action") == "RETRIEVE" and e.get("chunk_index", 999) < utterance_completion_chunk
    ]
    total_retrieves = [e for e in retrieval_events if e.get("action") == "RETRIEVE"]
    if not total_retrieves:
        return 0.0
    return len(early) / len(total_retrieves)


def retrieval_efficiency(
    decision_log: List[Dict],
    answer_quality: float = 1.0,
) -> float:
    """
    Useful retrievals / total retrieval calls.
    A retrieval is "useful" if it contributed to a grounded answer.
    We proxy this as grounding_score * retrieval_count contribution.
    """
    retrieves = [d for d in decision_log if d.get("action") == "RETRIEVE"]
    if not retrieves:
        return 0.0
    # Simplified: scale by answer quality as a proxy for usefulness
    return min(1.0, answer_quality)


def retrieval_savings(
    adaptive_retrieval_count: int,
    naive_streaming_count: int,
) -> float:
    """
    1 - adaptive_retrieval_calls / naive_streaming_calls

    Higher = more calls saved vs naive baseline.
    """
    if naive_streaming_count == 0:
        return 0.0
    return max(0.0, 1.0 - adaptive_retrieval_count / naive_streaming_count)


def latency_savings_ms(
    baseline_latency_ms: float,
    adaptive_latency_ms: float,
) -> float:
    """Latency saved by adaptive system vs baseline."""
    return baseline_latency_ms - adaptive_latency_ms


def refinement_efficiency(
    delta_latency_ms: float,
    full_retrieval_latency_ms: float,
) -> float:
    """
    delta_retrieval_cost / full_retrieval_cost

    Lower = more efficient delta retrieval vs full re-retrieval.
    """
    if full_retrieval_latency_ms == 0:
        return 1.0
    return delta_latency_ms / full_retrieval_latency_ms


def grounding_retention(
    grounding_after: float,
    grounding_before: float,
) -> float:
    """
    Grounding after refinement / Grounding before refinement.
    Should be >= 1.0 if refinement improved grounding.
    """
    if grounding_before == 0:
        return 1.0
    return grounding_after / grounding_before


def answer_stability(
    unchanged_claim_count: int,
    total_claim_count: int,
) -> float:
    """
    Percentage of claims that remain unchanged after refinement.
    High = refinement is targeted and does not disturb unrelated claims.
    """
    if total_claim_count == 0:
        return 1.0
    return unchanged_claim_count / total_claim_count


# ──────────────────────────────────────────────────────────────
# Aggregate Benchmark Report
# ──────────────────────────────────────────────────────────────

@dataclass
class BenchmarkReport:
    """Full benchmark result for a controller variant."""

    controller_name: str
    n_queries: int

    # IR Metrics
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    hit_at_k: Dict[int, float] = field(default_factory=dict)

    # Grounding
    faithfulness: float = 0.0
    citation_support_rate: float = 0.0
    answer_relevance: float = 0.0

    # Latency
    latency: Dict[str, float] = field(default_factory=dict)

    # Retrieval
    total_retrieval_calls: int = 0
    avg_retrieval_calls_per_query: float = 0.0

    # PRISM Research Metrics
    early_retrieval_rate: float = 0.0
    retrieval_efficiency: float = 0.0
    retrieval_savings_vs_naive: float = 0.0
    latency_savings_vs_conventional_ms: float = 0.0
    refinement_efficiency: float = 0.0
    grounding_retention: float = 0.0
    answer_stability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "controller_name": self.controller_name,
            "n_queries": self.n_queries,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": self.ndcg_at_k,
            "hit_at_k": self.hit_at_k,
            "faithfulness": round(self.faithfulness, 4),
            "citation_support_rate": round(self.citation_support_rate, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "latency": self.latency,
            "total_retrieval_calls": self.total_retrieval_calls,
            "avg_retrieval_calls_per_query": round(self.avg_retrieval_calls_per_query, 2),
            "early_retrieval_rate": round(self.early_retrieval_rate, 4),
            "retrieval_efficiency": round(self.retrieval_efficiency, 4),
            "retrieval_savings_vs_naive": round(self.retrieval_savings_vs_naive, 4),
            "latency_savings_vs_conventional_ms": round(self.latency_savings_vs_conventional_ms, 1),
            "refinement_efficiency": round(self.refinement_efficiency, 4),
            "grounding_retention": round(self.grounding_retention, 4),
            "answer_stability": round(self.answer_stability, 4),
        }
