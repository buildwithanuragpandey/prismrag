# Evaluation Results Directory Index

This directory contains the quantitative benchmark outputs, progressive ablation metrics, failure analysis records, and development diagnostic artifacts for the **Adaptive Agentic RAG** project.

---

## 1. Canonical Final Release Artifacts (Untouched Benchmark)

These authoritative artifacts document the frozen **V2-A** architecture evaluated on the **$N=100$ disjoint final untouched test set** (`evaluation/datasets/final_untouched_test.json`):

| Artifact File | Format | Description |
| :--- | :---: | :--- |
| [`final_metrics.json`](final_metrics.json) | JSON | Comprehensive metrics across all 5 comparative systems (Naive Dense, BM25, Hybrid RRF, Hybrid+Reranker, Adaptive Agentic). |
| [`final_evaluation_report.md`](final_evaluation_report.md) | Markdown | Formatted summary report of multi-system comparative benchmark results. |
| [`final_ablation_metrics.json`](final_ablation_metrics.json) | JSON | Cumulative 7-stage ablation metrics ($A0 \to A6$) isolating every pipeline subsystem. |
| [`final_ablation_report.md`](final_ablation_report.md) | Markdown | Formatted report on progressive subsystem ablations and transitions. |
| [`final_failure_analysis.json`](final_failure_analysis.json) | JSON | Diagnostic root-cause taxonomy across all 74 answerable failure cases. |
| [`final_failure_analysis.md`](final_failure_analysis.md) | Markdown | Formatted failure analysis breakdown (37.8% retrieval vs 62.2% downstream). |
| [`a6_consistency_validation.json`](a6_consistency_validation.json) | JSON | Audit verifying execution identity between comparative benchmark and ablation runners. |
| [`canonical_architecture_audit.json`](canonical_architecture_audit.json) | JSON | Provenance record verifying execution on canonical V2-A corpus and collection. |
| [`repository_release_readiness_audit.json`](repository_release_readiness_audit.json) | JSON | Full pre-release readiness audit record. |
| [`final_dataset_creation_report.md`](final_dataset_creation_report.md) | Markdown | Protocol and data isolation report for the untouched test set. |

---

## 2. Historical Subsystem Development & Diagnostic Records

These historical records document intermediate experimental milestones leading to the frozen V2-A architecture:

- **Chunking & Representation A/B Tests**: `dense_chunking_ab.json`, `dense_representation_ab.json`, `semantic_dilution_v1.json`.
- **Retrieval & Reranking Sweeps**: `full_retrieval_v1_v2.json`, `hybrid_rrf_full.json`, `hybrid_reranker_full.json`, `hybrid_reranker_mmr_full.json`, `mmr_lambda_sweep.json`.
- **Evidence Gating & Routing Calibration**: `evidence_gate_v2_500.json`, `evidence_gate_baseline_500.json`, `router_dev.json`, `router_test.json`.
- **Self-Correction & Semantic Rescue Policy**: `source_targeted_retry_ablation.json`, `query_rewriter_v2_ablation.json`, `safe_semantic_rescue_policy.json`, `strict_safe_semantic_rescue_policy.json`.
- **Grounding & Conclusion Verification**: `nli_grounding_modes_diagnostic.json`, `relevance_safety_floor_calibration.json`, `semantic_conclusion_ablation_results.json`.

