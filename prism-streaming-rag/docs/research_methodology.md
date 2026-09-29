# PRISM Research Methodology

**Project**: Adaptive Streaming RAG with Utility-Aware Retrieval and Incremental Evidence Refinement  
**Institution**: Samsung PRISM  
**Status**: Prototype / Experimental  

---

## 1. Research Question

> Can an adaptive retrieval controller — operating on partial, evolving queries — reduce unnecessary retrieval operations and response latency while maintaining answer quality, factual grounding, and citation correctness compared to conventional and naive streaming RAG baselines?

Secondary question:
> When new constraints arrive after an answer has been generated, can targeted delta retrieval recover factual grounding more efficiently than restarting the full RAG pipeline?

---

## 2. Hypotheses

**H1 (Primary)**: An adaptive utility-aware controller will execute fewer retrieval calls than naive streaming RAG, while achieving comparable or better answer quality and grounding.

**H2**: The adaptive controller will begin useful retrieval earlier than conventional RAG (before the utterance is complete), reducing time-to-first-answer.

**H3**: Delta retrieval will achieve higher refinement efficiency (lower cost) than full pipeline restart when new constraints arrive, while preserving grounding quality.

**H4**: Multi-intent decomposition + parallel retrieval will outperform sequential single-query retrieval on compound queries.

---

## 3. Baselines

| Name | Description |
|------|-------------|
| **Conventional RAG** | Wait for complete query → single retrieve → generate |
| **Naive Streaming RAG** | Retrieve on every transcript chunk |
| **Fixed Threshold RAG** | Retrieve when completeness > fixed threshold |
| **Adaptive PRISM (Ours)** | Utility-aware controller with all signals |

---

## 4. Independent Variables (Ablations)

| Ablation | What is removed |
|----------|----------------|
| No Semantic Novelty | Novelty score = 0 always |
| No Intent Stability | Stability score = 0 always |
| No Evidence Coverage | Evidence gap = 1 always |
| No Retrieval Utility | Replace with fixed threshold |
| No Multi-Intent | Single query per chunk only |
| No Delta Retrieval | Full re-retrieval on constraints |

---

## 5. Dependent Variables (Metrics)

### Standard IR
- Recall@K (K=1,3,5,10)
- Precision@K
- MRR
- nDCG@10
- Hit@K

### Faithfulness
- Claim grounding ratio (token overlap, upgradeable to NLI)
- Citation support rate (inline citations → valid evidence)

### Latency
- Mean total latency
- P50, P95 latency
- Time-to-first-retrieval (streaming specific)

### PRISM Research Metrics
- **Early Retrieval Rate**: % queries where retrieval starts before utterance complete
- **Retrieval Efficiency**: useful retrievals / total retrievals
- **Retrieval Savings**: 1 - adaptive_calls / naive_calls
- **Latency Savings**: baseline_latency - adaptive_latency
- **Refinement Efficiency**: delta_retrieval_cost / full_retrieval_cost
- **Grounding Retention**: grounding_after / grounding_before refinement
- **Answer Stability**: unchanged_claims / total_claims after refinement

---

## 6. Evaluation Gates (G1-G6)

| Gate | Criterion | Threshold |
|------|-----------|-----------|
| G1 Reproducibility | Same decisions on re-run | ≥ 95% match |
| G2 Early Retrieval | Retrieval before final chunk | ≥ 30% queries |
| G3 Multi-Intent | Correct intent identification | F1 ≥ 0.50 |
| G4 Factual Grounding | Claims grounded in evidence | ≥ 70% claims |
| G5 Session Refinement | Delta retrieval on constraint | ≥ 80% delta rate |
| G6 Telemetry | All event types logged | 100% coverage |

---

## 7. Dataset

- **Primary**: MultiHopRAG (yixuantt/MultiHopRAG)
- **Corpus**: 8,173 chunks (1000 char, 100 overlap) from adaptive-agentic-rag V2-A
- **Streaming simulation**: Utterance fragments generated from gold questions
- **Test set**: 100 held-out questions (final_untouched_test.json — disjoint from dev)

> [!IMPORTANT]
> No benchmark answers are hardcoded in system logic.
> The test set is never used during development or calibration.

---

## 8. Controller Policy

The initial controller uses a **transparent weighted linear policy**:

```
utility = w1 * novelty
         + w2 * intent_stability
         + w3 * completeness
         + w4 * evidence_gap
         - w5 * retrieval_cost
```

Default weights: `w1=0.30, w2=0.25, w3=0.20, w4=0.20, w5=0.05`

**This is a research baseline — not a scientifically optimized formula.**

The policy is modular (UtilityPolicy interface) so a learned policy can be evaluated later.

---

## 9. Experimental Protocol

1. Ingest corpus → build FAISS index + BM25 index
2. For each test query:
   a. Simulate streaming as 3-4 progressive fragments
   b. Run adaptive controller + retrieval
   c. Record all decisions and metrics
3. Re-run identical settings → verify G1 (reproducibility)
4. Run baseline controllers on same queries
5. For refinement test: add constraint after V1 answer
6. Compute all metrics and run G1-G6 gates

---

## 10. Limitations

- The query completeness estimator is a heuristic; a more sophisticated NLP-based estimator would improve accuracy.
- The claim grounding checker uses token overlap; NLI-based grounding (DeBERTa-v3) would be more rigorous.
- The intent classifier defaults to regex patterns; LLM-based decomposition requires a running LLM server.
- Streaming simulation generates artificial chunks from full queries; real ASR input would produce noisier fragments.
- All weights are set heuristically; formal optimization (Bayesian tuning) is future work.

---

## 11. Results

> [!WARNING]
> Experimental results have not yet been generated.
> This section will be populated after running the full benchmark suite.
> Do NOT populate this section with fabricated numbers.

Results will be written to `results/benchmark/` after running:
```
make benchmark
```

---

## 12. Ablation Protocol

Ablation studies follow this protocol:

1. Establish full adaptive system performance (baseline)
2. For each ablation: zero out or disable one signal component
3. Re-run full evaluation on same test set
4. Record delta in all metrics vs. full system
5. Repeat 3 times for statistical stability

Results go to `results/ablation/`.
