# Adaptive Agentic RAG v1.0.0 — Release Notes

> **Canonical V2-A Baseline Release**: A production-oriented reference implementation of multi-hop agentic RAG featuring hybrid retrieval, cross-encoder reranking, evidence-aware retry/recovery, single-pass grounded generation, fail-closed semantic safety verification, and an interactive visual showcase.

---

## Release Highlights

- **Canonical V2-A Pipeline**: Frozen multi-stage architecture integrating dense semantic embeddings (Qwen-0.6B) and sparse lexical signals (BM25Okapi) via Reciprocal Rank Fusion ($k=60$), cross-attention reranking (BGE-reranker-base), and Maximal Marginal Relevance diversity filtering ($\lambda=0.7$, top 5).
- **Evidence Gating & Adaptive Recovery**: Structural entity anchor and publisher availability checks prevent unwarranted generation; source-targeted query rewriting and semantic rescue recover multi-hop document links.
- **NLI Claim Grounding & Fail-Closed Semantic Safety**: Atomic proposition extraction with DeBERTa-v3 NLI verification ($P \ge 0.70$) and strict conclusion verification, trading answer coverage to suppress ungrounded or speculative assertions to `UNKNOWN`.
- **Production-Oriented FastAPI Service**: Non-blocking asynchronous query processing, serial inference execution, verified citation resolution, and structured trace telemetry.
- **Visual Web Showcase**: Standalone dark-themed engineering dashboard (`demo/`) demonstrating the 7-node LangGraph execution flow with live latency metrics and citation cards.
- **Portable CI & Regression Suite**: GitHub Actions workflow running 121 portable unit/integration tests under Python 3.12 without GPU/database dependencies; full local suite passing with 179/179 tests.

---

## Architecture Specification (Canonical V2-A)

```text
Query
  │
  ├──► 1. Query Router (Simple vs. Complex Route Selection)
  │
  ├──► 2. Hybrid Retrieval (Dense Qwen3-0.6B + Sparse BM25Okapi, RRF k=60)
  │
  ├──► 3. Cross-Encoder Reranking (BAAI/bge-reranker-base + MMR λ=0.7, top-5)
  │
  ├──► 4. Evidence Gating & Adaptive Recovery
  │      ├── Explicit Source Coverage Guard
  │      ├── Corpus Source Availability Check
  │      └── Source-Targeted Rewrite & Semantic Rescue (Max 1 retry)
  │
  ├──► 5. Single-Pass Structured Generation (Qwen2.5-1.5B-Instruct ChatML)
  │
  ├──► 6. Propositional Claim Grounding (DeBERTa-v3-small NLI, P ≥ 0.70)
  │
  └──► 7. Semantic Safety Verifier (StructuredConclusionVerifier)
         ├── Supported: Verified Direct Answer + Grounded Citations
         └── Unsupported: Fail-Closed Safe Abstention (UNKNOWN)
```

---

## Authoritative Final Benchmark Evaluation

Evaluated against the isolated, untouched `final_untouched_test.json` benchmark (100 multi-hop queries: 86 answerable multi-hop questions, 14 unanswerable null questions):

### Canonical Full-System Metrics (A6)

| Metric Category | Metric Name | Canonical A6 Value | Scope / Definition |
| :--- | :--- | :---: | :--- |
| **Retrieval Quality** | **Recall@10** | **0.866** (86.6%) | Multi-hop gold passage retrieval in top 10 candidates |
| | **MRR@10** | **0.757** | Mean Reciprocal Rank of first relevant passage |
| | **nDCG@10** | **0.729** | Normalized Discounted Cumulative Gain at rank 10 |
| **Safety & Abstention** | **Null Abstention Rate** | **92.9%** (13 / 14) | Safe abstention on unanswerable / out-of-corpus queries |
| | **Citation Validity** | **100.0%** | All generated citations map to valid retrieved corpus passages |
| | **Post-Verifier Citation Precision** | **87.0%** (20 / 23) | Cited passages matching ground-truth gold evidence post-verification |
| | **Pre-Verifier Citation Precision** | **88.5%** (23 / 26) | Evidence alignment on pre-verifier generated answer scope |
| **Answer Performance** | **Answer Coverage** | **31.4%** (27 / 86) | Percentage of answerable queries where pipeline committed to an answer |
| | **False Abstention Rate** | **68.6%** (59 / 86) | Conservative abstention on answerable queries with partial evidence |
| | **Answered Accuracy** | **44.4%** (12 / 27) | Exact proposition accuracy on answered queries |
| | **Overall Answerable Accuracy** | **14.0%** (12 / 86) | Global proposition accuracy across all 86 answerable queries |
| **Latency Profile** | **Mean Pipeline Latency** | **3.00s – 3.33s** | End-to-end CPU inference runtime (A6 ablation: 3.00s, Final Eval: 3.33s) |

---

## Baseline Progression Highlights

Empirical progression across frozen architectures on the final benchmark (`evaluation/results/final_metrics.json`):

| Pipeline Stage | Recall@10 | nDCG@10 | Null Abstention | Answered Accuracy | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Dense RAG** | 0.731 | 0.620 | 64.3% | 48.3% (28/58) | 32.6% (28/86) |
| **BM25 RAG** | 0.743 | 0.673 | 78.6% | 40.4% (23/57) | 26.7% (23/86) |
| **Hybrid RAG (Dense + BM25)** | 0.782 | 0.692 | 71.4% | 43.9% (29/66) | 33.7% (29/86) |
| **Hybrid + Cross-Encoder Reranker** | 0.842 | 0.711 | 78.6% | 51.8% (29/56) | 33.7% (29/86) |
| **Adaptive Agentic RAG (Canonical A6)** | **0.866** | **0.729** | **92.9%** | **44.4% (12/27)** | **14.0% (12/86)** |

> **Architectural Insight**: Hybrid retrieval and cross-encoder reranking provide strong retrieval gains (Recall@10 improving from 0.731 to 0.842, and to 0.866 with adaptive recovery). Downstream evidence gating and semantic safety verification prioritize precision and safe abstention (Null Abstention: 92.9%, Citation Validity: 100%) over raw answer coverage.

---

## Production-Oriented FastAPI Service & Web Showcase

- **REST Endpoints**:
  - `GET /health`: Fast liveness check (`{"status": "ok"}`).
  - `GET /ready`: Pre-flight readiness check verifying model weights and Qdrant collection status.
  - `GET /v1/system`: System configuration metadata, chunk counts (8,173), and architecture version.
  - `POST /v1/query`: Core query interface supporting direct answers, formatted text, grounded citations, and optional execution trace telemetry (`include_trace=true`).
- **Web Showcase (`demo/`)**: Pure HTML/CSS/JS interface providing live interactive queries, curated multi-hop presets, and visual tracking across all 7 pipeline stages.

---

## CI & Automated Test Coverage

- **Portable GitHub Actions CI**: Runs 121 unit and integration tests under Python 3.12 with `uv sync --frozen` on every push/PR without requiring GPU or local Qdrant services.
- **Full Local Test Regression**: 179 passed / 0 failed across vector store, hybrid retrieval, reranker, evidence gate, claim grounder, generator, and API layers.

---

## Known Limitations & Design Trade-offs

1. **Conservative Abstention Floor**: The fail-closed semantic verifier converts uncertain propositions to `UNKNOWN`, achieving high safety (92.9% null abstention) at the cost of answer coverage (31.4% on complex multi-hop questions).
2. **Multi-Hop Compositional Reasoning**: Multi-hop queries requiring synthesis across $>2$ disjoint publisher domains can fail evidence gating if bridging context receives low cross-attention reranking scores.
3. **Domain Scope**: Index calibrations and prompt representations are tuned for multi-source news synthesis (MultiHopRAG); specialized domains (biomedical, legal) require domain-specific indexing and re-calibration.

---

## Documentation Index

- [`README.md`](../README.md): Project overview, quickstart guide, and architectural diagrams.
- [`docs/final_technical_report.md`](final_technical_report.md): Formal research report detailing mathematical formulations, ablation metrics, and failure taxonomy.
- [`docs/system_implementation_guide.md`](system_implementation_guide.md): Module-by-module technical architecture and design guide.
- [`docs/api.md`](api.md): REST API specification and request/response contracts.
- [`docs/demo.md`](demo.md): Visual web showcase and CLI testing guide.
- [`docs/canonical_architecture_manifest.md`](canonical_architecture_manifest.md): Invariant specification of the frozen V2-A architecture.

---

## Reproducibility & Getting Started

```powershell
# 1. Clone repository
git clone https://github.com/mmnsrti/adaptive-agentic-rag.git
cd adaptive-agentic-rag

# 2. Setup virtual environment with uv
uv venv .venv --python 3.12
.venv\Scripts\activate
uv sync --frozen

# 3. Start the FastAPI service
uvicorn adaptive_agentic_rag.api.app:create_app --factory --host 127.0.0.1 --port 8000

# 4. Run automated test suite
python -m pytest -q
```
