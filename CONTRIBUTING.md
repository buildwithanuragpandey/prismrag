# Contributing to Adaptive Agentic RAG

Thank you for your interest in contributing to **Adaptive Agentic RAG**. This document outlines the development workflow, architecture guidelines, evaluation integrity principles, and pull request procedures.

---

## Development Setup

The project uses **Python 3.12** and **[`uv`](https://github.com/astral-sh/uv)** for fast, deterministic dependency management.

### Prerequisites
- Python 3.12+
- `uv` package manager

### Getting Started

```powershell
# 1. Clone the repository
git clone https://github.com/mmnsrti/adaptive-agentic-rag.git
cd adaptive-agentic-rag

# 2. Create and activate a virtual environment
uv venv .venv --python 3.12
.venv\Scripts\activate

# 3. Install locked dependencies
uv sync --frozen
```

---

## Repository Structure

```text
adaptive-agentic-rag/
├── data/                             # Processed chunk corpus and vector data
├── demo/                             # Standalone visual web showcase
├── docs/                             # Architecture manifests, technical reports, guides
├── evaluation/                       # Evaluation runners and benchmark datasets
│   ├── datasets/final_untouched_test.json # 100 isolated untouched test cases (PROTECTED)
│   └── results/                      # Authoritative benchmark JSON outputs
├── scripts/                          # Indexing, demo, and verification scripts
├── src/adaptive_agentic_rag/         # Core system source code
│   ├── agents/                       # Query router, evidence grader, rewriter
│   ├── api/                          # FastAPI application, routes, schemas, service
│   ├── embeddings/                   # Qwen embedding model wrapper
│   ├── generation/                   # Generator, claim grounder, semantic verifier
│   ├── orchestration/                # LangGraph state machine and retry policies
│   ├── processing/                   # Chunker and cleaner
│   ├── retrieval/                    # Dense, BM25, Hybrid RRF, Cross-Encoder reranker
│   └── vectorstore/                  # Qdrant client integration
└── tests/                            # 179 automated unit, integration, and API tests
```

---

## Architecture Integrity (Canonical V2-A)

The core RAG pipeline is standardized on the **V2-A (Frozen Canonical)** architecture. 

When contributing modifications to core modules:
1. **Explicit Rationale**: Any changes to retrieval algorithms, reranking parameters, evidence thresholds, or generation prompts must have a documented rationale.
2. **Safety & Coverage Discipline**: Modifications must evaluate trade-offs between answer coverage and fail-closed safety (abstention). Do not disable safety gating to artificially inflate coverage.
3. **Reproducibility**: Parameter adjustments must be verified against development subsets before proposing changes.

---

## Evaluation & Benchmark Integrity

To maintain rigorous scientific standards:
- **`evaluation/datasets/final_untouched_test.json` is Strictly Read-Only**: Never tune thresholds, modify prompts, or select demo queries based on the final test set.
- **Strict Partitioning**: Use separate development datasets (`evaluation/datasets/frozen_eval_500.json`) for exploration and calibration.
- **Historical Reports**: Do not alter historical evaluation records in `docs/final_technical_report.md` unless correcting documented typos.

---

## Testing Guidelines

All contributions must pass the automated test suite before merging.

### 1. CI-Safe Test Suite (Portable)
GitHub Actions runs the 121 portable unit and integration tests defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) on every push and PR:

```powershell
# Run CI-safe subset locally
pytest tests/test_adaptive_retry_policy.py tests/test_explicit_source_coverage.py tests/test_structured_conclusion_verifier.py tests/test_api_query.py -q
```

### 2. Full Local Regression Suite
The complete 179-test suite validates full model pipelines, NLI entailment, and vector search:

```powershell
# Run full test regression
python -m pytest -q
```

---

## Code Quality Standards

- **Separation of Concerns**: Keep API routes, orchestration graphs, and retriever implementations modular.
- **Type Annotations**: Use type hints across all function signatures and Pydantic v2 schemas for API contracts.
- **Deterministic Logic**: Ensure routing, scoring, and citation mappings are reproducible and thread-safe.
- **No Chain-of-Thought in Traces**: Keep public API response traces focused on execution metrics, candidate counts, and citation validation telemetry.

---

## Demo & Showcase Guidelines

When modifying the visual web demo (`demo/`):
- **Real API Communication**: The demo must connect to live FastAPI endpoints (`/v1/query`, `/health`, `/ready`, `/v1/system`).
- **No Mock Inference**: Never hardcode simulated answers or synthetic citation badges.
- **Disjoint Demo Queries**: Ensure preset scenario queries do not overlap with the final untouched benchmark test set.
- **Accurate Telemetry**: Clearly differentiate client HTTP round-trip latency from benchmark model execution time.

---

## Pull Request Guidelines

1. **Branch Naming**: Use descriptive branch names (e.g., `feature/custom-retriever`, `fix/api-cors-headers`, `docs/setup-clarification`).
2. **PR Description**: Include a summary explaining:
   - What changed and why.
   - How changes were verified (test commands run).
   - Any architectural or latency impacts.
3. **Clean Diffs**: Avoid committing temporary debug scripts, `.log` files, or cache directories.

---

## Commit Message Convention

We follow standard Conventional Commits:

- `feat:` New features or pipeline capabilities.
- `fix:` Bug fixes in retrieval, generation, or API layers.
- `docs:` Documentation improvements or report additions.
- `test:` Adding or updating unit/integration tests.
- `ci:` GitHub Actions workflow updates.
- `chore:` Dependency updates, metadata alignments, or repository maintenance.

---

## Security Issues

For potential security vulnerabilities, please refer to our [Security Policy](SECURITY.md) for confidential reporting procedures.

