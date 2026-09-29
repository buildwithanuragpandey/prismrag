"""
PRISM Hybrid Retrieval Engine (Phase 9, 12)

Wraps the existing adaptive-agentic-rag retrieval infrastructure:
  - BM25 sparse retrieval
  - Dense vector retrieval (FAISS)
  - RRF fusion
  - Cross-encoder reranking
  - MMR diversity filtering
  - Parallel sub-query retrieval for multi-intent decomposition

Interfaces are modular so any component can be swapped independently.

NOTE: We reuse the battle-tested components from adaptive-agentic-rag.
      This file provides the PRISM-specific orchestration layer on top.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Add adaptive-agentic-rag source to path so we can import its components
_AAR_SRC = Path(__file__).parents[5] / "src"
if str(_AAR_SRC) not in sys.path:
    sys.path.insert(0, str(_AAR_SRC))


# ──────────────────────────────────────────────────────────────
# Result Data Model
# ──────────────────────────────────────────────────────────────

class RetrievedChunk:
    """A single retrieved document chunk."""

    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        text: str,
        score: float,
        rerank_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.text = text
        self.score = score
        self.rerank_score = rerank_score
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text[:500],    # Truncate for API responses
            "score": round(self.score, 4),
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score else None,
            "source": self.metadata.get("source", ""),
            "title": self.metadata.get("title", ""),
            "url": self.metadata.get("url", ""),
        }


class RetrievalResult:
    """Result of a single retrieval call."""

    def __init__(
        self,
        retrieval_id: str,
        query: str,
        chunks: List[RetrievedChunk],
        latency_ms: float,
        method: str = "hybrid",
        sub_query_results: Optional[Dict[str, List[RetrievedChunk]]] = None,
    ) -> None:
        self.retrieval_id = retrieval_id
        self.query = query
        self.chunks = chunks
        self.latency_ms = latency_ms
        self.method = method
        self.sub_query_results = sub_query_results or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_id": self.retrieval_id,
            "query": self.query,
            "chunk_count": len(self.chunks),
            "latency_ms": round(self.latency_ms, 1),
            "method": self.method,
            "chunks": [c.to_dict() for c in self.chunks],
        }


# ──────────────────────────────────────────────────────────────
# FAISS-based Dense Retriever (self-contained)
# ──────────────────────────────────────────────────────────────

class FAISSDenseRetriever:
    """
    Dense retriever backed by a FAISS flat index.
    Loads an existing index and metadata JSON from disk.

    Falls back to BM25 if FAISS is not available or index not built.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        embedding_fn,
        top_k: int = 20,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_fn = embedding_fn
        self.top_k = top_k
        self._index = None
        self._metadata: List[Dict] = []

    def _load(self):
        if self._index is not None:
            return True

        try:
            import faiss, json
            if not self.index_path.exists():
                logger.warning("FAISS index not found at %s", self.index_path)
                return False
            self._index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info("FAISS index loaded: %d vectors", self._index.ntotal)
            return True
        except ImportError:
            logger.warning("faiss-cpu not installed. Dense retrieval unavailable.")
            return False
        except Exception as exc:
            logger.warning("Failed to load FAISS index: %s", exc)
            return False

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        if not self._load():
            return []

        import numpy as np
        k = top_k or self.top_k
        emb = self.embedding_fn(query)
        query_vec = np.array([emb], dtype=np.float32)

        distances, indices = self._index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            doc = self._metadata[idx].copy()
            # Convert L2 distance to pseudo-similarity score
            doc["_score"] = float(1.0 / (1.0 + dist))
            results.append(doc)

        return results


# ──────────────────────────────────────────────────────────────
# PRISM Hybrid Retriever
# ──────────────────────────────────────────────────────────────

class PRISMHybridRetriever:
    """
    PRISM's hybrid retrieval engine.

    Combines:
      - BM25 sparse retrieval (from adaptive-agentic-rag)
      - FAISS dense retrieval
      - RRF fusion (from adaptive-agentic-rag)
      - BGE cross-encoder reranking (from adaptive-agentic-rag)

    Also supports:
      - Parallel sub-query retrieval for multi-intent decomposition
      - Result deduplication

    Args:
        corpus_path: Path to processed_corpus JSON
        embedding_fn: Callable for dense embeddings
        faiss_index_path: FAISS index file
        faiss_metadata_path: FAISS metadata JSON
        top_k_bm25: BM25 candidates
        top_k_dense: Dense candidates
        top_k_rrf: After RRF fusion
        top_k_reranked: After reranking
        rerank_enabled: Enable cross-encoder reranking
        reranker_model: HuggingFace model for reranking
    """

    def __init__(
        self,
        corpus_path: str,
        embedding_fn,
        faiss_index_path: str = "data/faiss_index/index.faiss",
        faiss_metadata_path: str = "data/faiss_index/metadata.json",
        top_k_bm25: int = 20,
        top_k_dense: int = 20,
        top_k_rrf: int = 20,
        top_k_reranked: int = 5,
        rerank_enabled: bool = True,
        reranker_model: str = "BAAI/bge-reranker-base",
        rrf_k: int = 60,
    ) -> None:
        self.embedding_fn = embedding_fn
        self.top_k_bm25 = top_k_bm25
        self.top_k_dense = top_k_dense
        self.top_k_rrf = top_k_rrf
        self.top_k_reranked = top_k_reranked
        self.rerank_enabled = rerank_enabled
        self.reranker_model = reranker_model
        self.rrf_k = rrf_k

        # BM25 (from adaptive-agentic-rag)
        self._bm25: Optional[Any] = None
        self._corpus_path = corpus_path

        # Dense retriever
        self._dense = FAISSDenseRetriever(
            index_path=faiss_index_path,
            metadata_path=faiss_metadata_path,
            embedding_fn=embedding_fn,
            top_k=top_k_dense,
        )

        # Reranker (lazy load)
        self._reranker = None

    def _load_bm25(self):
        """Lazy load BM25 from corpus."""
        if self._bm25 is not None:
            return

        try:
            # Try to import from adaptive-agentic-rag
            from adaptive_agentic_rag.retrieval.bm25_retriever import BM25Retriever
            self._bm25 = BM25Retriever(corpus_path=self._corpus_path)
            logger.info("BM25 loaded from adaptive-agentic-rag")
        except ImportError:
            # Fall back to self-contained BM25
            logger.info("adaptive-agentic-rag not importable. Using self-contained BM25.")
            self._bm25 = self._build_fallback_bm25()

    def _build_fallback_bm25(self):
        """Build a minimal BM25 retriever without adaptive-agentic-rag."""
        import json
        from rank_bm25 import BM25Okapi

        corpus_path = Path(self._corpus_path)
        if not corpus_path.exists():
            logger.warning("Corpus not found at %s — BM25 unavailable", corpus_path)
            return None

        with open(corpus_path, encoding="utf-8") as f:
            documents = json.load(f)

        texts = []
        for doc in documents:
            content = doc.get("content", doc.get("text", ""))
            title = doc.get("title", "")
            source = doc.get("source", "")
            combined = f"{title} {source} {content}"
            texts.append(combined.lower().split())

        class SimpleBM25:
            def __init__(self, documents, texts):
                self.documents = documents
                self.bm25 = BM25Okapi(texts)

            def search(self, query: str, top_k: int = 20) -> List[Dict]:
                tokens = query.lower().split()
                scores = self.bm25.get_scores(tokens)
                indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
                results = []
                for idx in indices:
                    if scores[idx] > 0:
                        doc = self.documents[idx].copy()
                        doc["_score"] = float(scores[idx])
                        doc["id"] = doc.get("id", str(idx))
                        results.append(doc)
                return results

        return SimpleBM25(documents, texts)

    def _load_reranker(self):
        """Lazy load cross-encoder reranker."""
        if self._reranker is not None:
            return True
        if not self.rerank_enabled:
            return False

        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker: %s", self.reranker_model)
            self._reranker = CrossEncoder(self.reranker_model)
            return True
        except Exception as exc:
            logger.warning("Reranker load failed: %s. Reranking disabled.", exc)
            self.rerank_enabled = False
            return False

    def _rrf_fuse(
        self, result_lists: List[List[Dict]], k: int = 60
    ) -> List[Dict]:
        """Reciprocal Rank Fusion — reuse logic from adaptive-agentic-rag rrf.py."""
        try:
            from adaptive_agentic_rag.retrieval.rrf import reciprocal_rank_fusion
            return reciprocal_rank_fusion(result_lists, top_k=self.top_k_rrf, k=k)
        except ImportError:
            pass

        # Self-contained RRF fallback
        from collections import defaultdict
        scores: dict = defaultdict(float)
        documents = {}
        for results in result_lists:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc.get("id", str(rank))
                scores[doc_id] += 1.0 / (k + rank)
                if doc_id not in documents:
                    documents[doc_id] = doc.copy()

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fused = []
        for doc_id in sorted_ids[: self.top_k_rrf]:
            doc = documents[doc_id].copy()
            doc["_rrf_score"] = scores[doc_id]
            fused.append(doc)
        return fused

    def _rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank candidates using cross-encoder."""
        if not self._load_reranker() or not candidates:
            return candidates

        try:
            texts = [
                (query, c.get("content", c.get("text", "")))
                for c in candidates
            ]
            scores = self._reranker.predict(texts)
            for c, s in zip(candidates, scores):
                c["_rerank_score"] = float(s)
            candidates.sort(key=lambda c: c.get("_rerank_score", 0), reverse=True)
            return candidates[: self.top_k_reranked]
        except Exception as exc:
            logger.warning("Reranking failed: %s", exc)
            return candidates[: self.top_k_reranked]

    def _docs_to_chunks(
        self, docs: List[Dict], reranked: bool = False
    ) -> List[RetrievedChunk]:
        chunks = []
        for doc in docs:
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(doc.get("id", doc.get("chunk_id", ""))),
                    document_id=str(doc.get("doc_id", doc.get("document_id", ""))),
                    text=doc.get("content", doc.get("text", "")),
                    score=float(doc.get("_rrf_score", doc.get("_score", 0.0))),
                    rerank_score=float(doc.get("_rerank_score", 0.0)) if reranked else None,
                    metadata={
                        "source": doc.get("source", ""),
                        "title": doc.get("title", ""),
                        "url": doc.get("url", ""),
                    },
                )
            )
        return chunks

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve for a single query using hybrid RRF + reranking."""
        t_start = time.time()
        retrieval_id = str(uuid.uuid4())

        self._load_bm25()

        # BM25
        bm25_results = []
        if self._bm25 is not None:
            try:
                raw = self._bm25.search(query, top_k=self.top_k_bm25)
                # Normalize doc format
                for doc in raw:
                    if "id" not in doc:
                        doc["id"] = doc.get("chunk_id", str(id(doc)))
                bm25_results = raw
            except Exception as exc:
                logger.warning("BM25 failed: %s", exc)

        # Dense
        dense_results = self._dense.search(query, top_k=self.top_k_dense)
        for doc in dense_results:
            if "id" not in doc:
                doc["id"] = doc.get("chunk_id", str(id(doc)))

        # RRF Fusion
        candidates = self._rrf_fuse([bm25_results, dense_results])

        # Rerank
        if candidates:
            candidates = self._rerank(query, candidates)

        chunks = self._docs_to_chunks(candidates, reranked=self.rerank_enabled)
        latency_ms = (time.time() - t_start) * 1000

        return RetrievalResult(
            retrieval_id=retrieval_id,
            query=query,
            chunks=chunks,
            latency_ms=latency_ms,
            method="hybrid_rrf_reranked",
        )

    async def retrieve_async(self, query: str) -> RetrievalResult:
        """Async wrapper for retrieve()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, query)

    async def retrieve_parallel(
        self, sub_queries: List[str]
    ) -> Tuple[RetrievalResult, Dict[str, RetrievalResult]]:
        """
        Retrieve for multiple sub-queries in parallel.
        Returns merged result + per-subquery results.
        """
        if not sub_queries:
            return RetrievalResult("", "", [], 0.0), {}

        t_start = time.time()

        # Run all sub-queries concurrently
        tasks = [self.retrieve_async(q) for q in sub_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        sub_results: Dict[str, RetrievalResult] = {}
        all_chunks: Dict[str, RetrievedChunk] = {}

        for q, result in zip(sub_queries, results):
            if isinstance(result, Exception):
                logger.warning("Sub-query '%s' failed: %s", q, result)
                continue
            sub_results[q] = result
            for chunk in result.chunks:
                # Deduplicate by chunk_id
                if chunk.chunk_id not in all_chunks:
                    all_chunks[chunk.chunk_id] = chunk

        # Merge: sort by score descending
        merged_chunks = sorted(
            all_chunks.values(),
            key=lambda c: c.rerank_score or c.score,
            reverse=True,
        )

        merged = RetrievalResult(
            retrieval_id=str(uuid.uuid4()),
            query=" | ".join(sub_queries),
            chunks=merged_chunks[: self.top_k_reranked * 2],
            latency_ms=(time.time() - t_start) * 1000,
            method="parallel_hybrid_merged",
            sub_query_results={q: r.chunks for q, r in sub_results.items()},
        )

        return merged, sub_results
