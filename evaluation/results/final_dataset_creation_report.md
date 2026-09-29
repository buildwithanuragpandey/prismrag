# Final Untouched Test Set Creation Report

- **Dataset Path**: `evaluation\datasets\final_untouched_test.json`
- **Source Repository**: `yixuantt/MultiHopRAG`
- **Pinned Revision**: `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`
- **Selection Seed**: `2026`
- **Creation Timestamp**: `2026-09-01T13:49:17.081618+00:00`

---

## 1. Dataset Summary & Quotas

- **Total Selected Examples**: **100**
- **Answerable Cases**: **86** (86.0%)
- **Null Cases (Insufficient Information)**: **14** (14.0%)

### Question Type Breakdown
| Question Type | Count | Percentage |
| :--- | :---: | :---: |
| `inference_query` | 29 | 29.0% |
| `comparison_query` | 29 | 29.0% |
| `temporal_query` | 28 | 28.0% |
| `null_query` | 14 | 14.0% |

---

## 2. Disjointness & Isolation Guarantee

- **Excluded Previously Used Queries**: 620 (from `frozen_eval_500.json`, `frozen_e2e_smoke_20.json`, router sets, and gate splits).
- **Overlap Verification**: Exactly **0** questions in this test set overlap with any training, calibration, validation, or debugging datasets.
- **Evidence Integrity**: 100% of gold evidence references in answerable queries map deterministically to document IDs in `data/processed/processed_corpus.json`.

---

## 3. Gold Source Distribution

Top gold sources represented across the selected multi-hop evaluation set:
| Source Name | Document References Count |
| :--- | :---: |
| TechCrunch | 43 |
| The Verge | 24 |
| Fortune | 21 |
| Sporting News | 14 |
| The Independent - Life and Style | 6 |
| The Age | 5 |
| CBSSports.com | 5 |
| Cnbc | World Business News Leader | 4 |
| Essentially Sports | 4 |
| The Roar | Sports Writers Blog | 4 |
| The Independent - Sports | 3 |
| Polygon | 2 |
| Engadget | 2 |
| The Sydney Morning Herald | 2 |
| FOX News - Lifestyle | 2 |

---

## 4. Benchmark Freeze Protocol

This dataset is **FROZEN**.
- It must **never** be used for prompt tuning, threshold calibration, or error-specific rule additions.
- It serves strictly as the independent out-of-sample benchmark for comparative system evaluation.
