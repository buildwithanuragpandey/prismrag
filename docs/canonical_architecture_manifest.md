# Canonical Architecture Manifest: Frozen Stack V2-A

- **Document Version**: 1.0.0
- **Status**: **FROZEN CANONICAL**
- **Date**: 2026-09-02
- **Corpus Version**: **V2-A (8,173 chunks)**

---

## 1. Executive Specification

This manifest is the authoritative specification for all models, hyperparameters, corpus artifacts, vector collections, and evaluation invariants in the **Adaptive Agentic RAG** system.

---

## 2. Core Corpus & Chunking Specification

| Parameter | Canonical Value | Verification Source |
| :--- | :--- | :--- |
| **Corpus File** | `data/processed/processed_corpus_v2.json` | 8,173 chunks across 15+ news publishers |
| **Chunk Size** | 1,000 characters | [`chunker.py`](../src/adaptive_agentic_rag/processing/chunker.py) |
| **Chunk Overlap** | 100 characters | [`chunker.py`](../src/adaptive_agentic_rag/processing/chunker.py) |
| **Minimum Words** | 20 words | Paragraph merging threshold |
| **Chunk ID Format** | `doc_{id:04d}_chunk_{index}` | Deterministic chunk identifier |
| **Vector DB Storage** | `data/qdrant/` | Qdrant local file storage |
| **Collection Name** | `multihop_chunks_v2` | 8,173 points, 1024-dim Cosine distance |

---

## 3. Subsystem Model & Hyperparameter Specification

### 3.1 Dense Semantic Embeddings
- **Model**: `Qwen/Qwen3-Embedding-0.6B`
- **Embedding Dimension**: 1024 floats
- **Normalization**: $L_2$ Unit Norm (`normalize_embeddings=True`)
- **Query Instruction**: `prompt_name="query"` (`Instruct: Given a web search query...`)
- **Document Encoding**: Raw text without prompt prefix

### 3.2 Sparse Lexical Retrieval (BM25)
- **Algorithm**: `BM25Okapi` (`rank-bm25`)
- **Document String Formulation**: `f"Title: {title} | Source: {source} | Content: {text}"`
- **Parameters**: $k_1 = 1.5$, $b = 0.75$

### 3.3 Rank Fusion (RRF)
- **Algorithm**: Reciprocal Rank Fusion
- **Smoothing Constant**: $k = 60$
- **Input Candidates**: Top-20 Dense + Top-20 BM25 $\to$ Top-20 Merged

### 3.4 Cross-Encoder Reranking & MMR
- **Reranker Backbone**: `BAAI/bge-reranker-base` (`AutoModelForSequenceClassification`)
- **Scoring**: Full sequence pair cross-attention logits
- **Maximal Marginal Relevance (MMR)**: $\lambda = 0.7$, selecting Top-5 chunks

### 3.5 Query Routing (`QueryRouter V2`)
- **Classification**: Heuristic linguistic classifier
- **Routes**:
  - Comparison / Multi-Entity / Temporal $\to$ `"complex"` (Hybrid + Reranker + MMR)
  - Single-Predicate Fact $\to$ `"simple"` (Fast Dense Top-10)

### 3.6 Evidence Gating & Source Coverage
- **EvidenceGrader V2**: Minimum 2 distinct query entity anchor matches
- **ExplicitSourceCoverageGuard**: Scans query for named publishers; blocks generation if missing from candidates
- **CorpusSourceAvailability**: Verifies publisher exists in `processed_corpus_v2.json` before triggering retry

### 3.7 Adaptive Recovery & Rewriting
- **Max Retry Budget**: 1 retry (`max_retries = 1`)
- **QueryRewriter V2**: Constraint-preserving entity reformulation with missing publisher anchor
- **Source-Targeted Search**: `BM25Retriever.search_by_sources()`

### 3.8 LLM Generation & Verification
- **LLM Backbone**: `Qwen/Qwen2.5-1.5B-Instruct` (Single-pass ChatML)
- **NLI Grounder**: `cross-encoder/nli-deberta-v3-small` ($P(\text{entailment}) \ge 0.70$)
- **Relevance Filter**: `RelevanceFilter V2` (Global top-2 query-relevant claims)
- **Semantic Verifier**: `StructuredConclusionVerifier` (Fail-closed suppression to `"UNKNOWN"`)

---

## 4. Benchmark & Data Isolation Invariants

- **Untouched Benchmark**: `evaluation/datasets/final_untouched_test.json` ($N=100$)
- **Dev / Validation Set**: `evaluation/datasets/frozen_eval_500.json` ($N=500$, strictly disjoint)
- **Verified Recall@10**: **0.866** (86.6%)
- **Verified Citation Validity**: **100.0%**
- **Verified Null Abstention**: **92.9%** (13/14)
- **Verified Mean Latency**: **3.00s**

