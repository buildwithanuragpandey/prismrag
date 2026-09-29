# Canonical Architecture Audit Report

- **Audit Date**: 2026-09-02
- **Audit Objective**: Verify runtime execution provenance and dataset isolation for the frozen canonical **V2-A** architecture.
- **Machine-Readable Artifact**: [`evaluation/results/canonical_architecture_audit.json`](../evaluation/results/canonical_architecture_audit.json)

---

## 1. Executive Summary

A comprehensive provenance audit was performed across all retrieval initialization code, evaluation runners, and benchmark artifacts. The audit confirms **Outcome A (100% Valid Provenance)**:
- All benchmark runners ([`run_final_evaluation.py`](../evaluation/run_final_evaluation.py), [`run_final_ablation.py`](../evaluation/run_final_ablation.py)) instantiate canonical `RAGNodes()`.
- `RAGNodes()` defaults to `AdaptiveRetriever()`, which explicitly injects `collection_name="multihop_chunks_v2"` and `bm25_corpus_path="data/processed/processed_corpus_v2.json"`.
- All quantitative benchmark results in `final_metrics.json` and `final_ablation_metrics.json` were produced strictly on the canonical **V2-A** corpus (8,173 chunks) and Qdrant collection `multihop_chunks_v2`.

---

## 2. Provenance Verification Matrix

| Subsystem Component | Audited Code Path | Parameter Inspected | Verified Value | Provenance Integrity |
| :--- | :--- | :--- | :--- | :---: |
| **Corpus File** | `src/.../adaptive_retriever.py:20` | `DEFAULT_BM25_CORPUS_PATH` | `data/processed/processed_corpus_v2.json` | **VERIFIED** |
| **Vector Store** | `src/.../adaptive_retriever.py:15` | `DEFAULT_DENSE_COLLECTION` | `multihop_chunks_v2` | **VERIFIED** |
| **Chunk Count** | `processed_corpus_v2.json` | Array length | 8,173 chunks | **VERIFIED** |
| **Evaluation Runner** | `evaluation/run_final_evaluation.py:17` | `RAGNodes()` instantiation | Uses AdaptiveRetriever defaults | **VERIFIED** |
| **Ablation Runner** | `evaluation/run_final_ablation.py:18` | `RAGNodes()` instantiation | Uses AdaptiveRetriever defaults | **VERIFIED** |
| **Consistency Validation** | `evaluation/verify_a6_consistency.py` | Benchmark vs Ablation A6 | 100/100 identical states | **VERIFIED** |
| **FastAPI System Route** | `src/.../api/service.py:242` | `qdrant_collection` | `multihop_chunks_v2` | **VERIFIED** |

---

## 3. Dataset Disjointness Verification

The final evaluation dataset `evaluation/datasets/final_untouched_test.json` ($N=100$) was verified against development datasets:
- **Dev Set (`frozen_eval_500.json`)**: 0 overlapping question IDs (Strictly Disjoint).
- **Smoke Set (`frozen_e2e_smoke_20.json`)**: 0 overlapping question IDs (Strictly Disjoint).
- **Gold Evidence Mapping**: 100% of gold evidence references in answerable queries map deterministically to document IDs in `data/processed/processed_corpus_v2.json`.

---

## 4. Audit Conclusion

The benchmark artifacts in `evaluation/results/` are authoritative, reproducible, and internally consistent with the canonical **V2-A** architecture.

