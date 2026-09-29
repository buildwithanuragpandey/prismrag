import json
from dataclasses import dataclass
from pathlib import Path

from adaptive_agentic_rag.generation.relevance_filter import (
    ClaimRelevanceFilter,
    RelevantClaim,
)
from adaptive_agentic_rag.generation.claim_grounder import (
    ClaimSupport,
    GroundingResult,
)
from adaptive_agentic_rag.orchestration.explicit_source_coverage import (
    ExplicitSourceCoverageGuard,
)


RESULTS_PATH = Path("evaluation/results/e2e_smoke_20_results.json")
CORPUS_PATH = Path("data/processed/processed_corpus_v2.json")
OUTPUT_PATH = Path("evaluation/results/relevance_filter_invariant_ablation.json")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class MockContextItem:
    citation_id: int
    source: str
    title: str = ""


@dataclass
class MockContext:
    items: list[MockContextItem]


def build_document_source_catalog():
    payload = load_json(CORPUS_PATH)
    records = payload if isinstance(payload, list) else payload.get("records", payload.get("chunks", payload.get("data", [])))
    catalog = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        doc_id = record.get("document_id")
        meta = record.get("metadata", {}) or {}
        source = meta.get("source") or record.get("source") or ""
        title = meta.get("title") or record.get("title") or ""
        if doc_id:
            catalog[str(doc_id)] = {
                "source": str(source).strip(),
                "title": str(title).strip(),
            }
    return catalog


def main():
    records = load_json(RESULTS_PATH)
    doc_catalog = build_document_source_catalog()

    # Instantiate relevance filter
    relevance_filter = ClaimRelevanceFilter(
        reranker=None,  # Not used for cached scoring logic if already scored
        max_relevant_claims=2,
    )

    print("=" * 80)
    print("RELEVANCE FILTER INVARIANT ABLATION: GLOBAL TOP-2 vs SOURCE-AWARE ADAPTIVE")
    print("=" * 80)

    # Compare on all records
    comparisons = []
    differences_found = 0

    for r in records:
        rec_id = r.get("id")
        query = r.get("question", "")
        q_type = r.get("question_type", "")
        abstained = r.get("abstained", False)
        cited_doc_ids = r.get("cited_document_ids", []) or []
        cited_ids = r.get("cited_ids", []) or []

        # Reconstruct mock context from cited documents
        context_items = []
        for idx, (cid, doc_id) in enumerate(zip(cited_ids, cited_doc_ids), start=1):
            src = doc_catalog.get(str(doc_id), {}).get("source", "")
            context_items.append(MockContextItem(citation_id=cid, source=src))

        context = MockContext(items=context_items)

        # Check required sources
        guard = ExplicitSourceCoverageGuard()
        cov_res = guard.check(query=query, context=context)

        # Let's see if there are multiple sources
        has_multi_source = len(cov_res.required_sources) >= 2

        comparisons.append({
            "id": rec_id,
            "question_type": q_type,
            "required_sources": cov_res.required_sources,
            "has_multi_source": has_multi_source,
            "abstained": abstained,
            "cited_doc_ids": cited_doc_ids,
        })

    print(f"Total evaluated records: {len(records)}")
    multi_source_count = sum(1 for c in comparisons if c["has_multi_source"])
    print(f"Records with multi-source query requirement (>= 2): {multi_source_count}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "total_records": len(records),
            "multi_source_records": multi_source_count,
            "comparisons": comparisons,
        }, f, indent=2)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
