# Adaptive Agentic RAG — v1.0.0 Release Checklist

- **Release Target**: `v1.0.0`
- **Canonical Architecture**: `V2-A (Frozen Canonical)`
- **Verification Date**: 2026-09-02
- **Status**: Published

---

## Pre-Release Verification Checklist

- [x] **Git Status Reviewed**: Working tree clean, zero untracked artifacts.
- [x] **Secrets Scan Clean**: Zero API keys, bearer tokens, or credentials exposed across repository.
- [x] **Private Docs Ignored**: `.local_docs/project_master_learning_guide.md` strictly ignored by `.gitignore`.
- [x] **README Links Valid**: 100% of internal Markdown links and image references verified (0 broken links).
- [x] **CI Workflow Valid**: `.github/workflows/ci.yml` configured for Python 3.12 with `uv sync --frozen`.
- [x] **CI-Safe Tests Pass**: 121 unit & integration tests passing in 20.52s.
- [x] **Full Regression Passes**: 179 tests passing with 0 failures in local execution environment.
- [x] **Demo Screenshots Verified**: `answered_query.png`, `trace_mode.png`, and `safe_abstention.png` generated and visually audited.
- [x] **Final Test Unchanged**: `evaluation/datasets/final_untouched_test.json` 100% untouched.
- [x] **Benchmark Artifacts Unchanged**: `final_metrics.json`, `final_ablation_metrics.json`, `final_failure_analysis.json` 100% untouched.
- [x] **Technical Report Protected**: `docs/final_technical_report.md` has 0 lines modified.
- [x] **License Present**: MIT License file populated and declared in `README.md`.
- [x] **Pyproject / Lockfile Synchronized**: `pyproject.toml` dependencies match `uv.lock` (`uv lock --check` passed).
- [x] **Release Notes Reviewed**: `docs/release_notes_v1.0.0.md` finalized and aligned with empirical metrics.

---

## Final Authorization & Execution Checklist

- [x] **Commit(s) Created**: Working tree release changes committed with conventional commit format.
- [x] **Tag Created**: Git annotated tag `v1.0.0` created on release commit (`21e9393`).
- [x] **Pushed**: Branch `master` and tag `v1.0.0` pushed to remote `origin`.
- [x] **GitHub Release Published**: GitHub release published with `docs/release_notes_v1.0.0.md` content.
