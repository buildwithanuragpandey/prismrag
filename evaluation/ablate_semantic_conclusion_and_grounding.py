import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from adaptive_agentic_rag.orchestration.explicit_source_coverage import (
    ExplicitSourceCoverageGuard,
)
from adaptive_agentic_rag.generation.relation_aware_answer_resolver import (
    RelationAwareAnswerResolver,
)


RESULTS_PATH = Path(
    "evaluation/results/"
    "e2e_smoke_20_results.json"
)

CORPUS_PATH = Path(
    "data/processed/"
    "processed_corpus_v2.json"
)

OUTPUT_PATH = Path(
    "evaluation/results/"
    "semantic_conclusion_ablation_results.json"
)


# ============================================================
# Loading
# ============================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_records():
    payload = load_json(RESULTS_PATH)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "examples", "data"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
    raise ValueError(f"Unsupported results structure: {type(payload).__name__}")


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


# ============================================================
# Normalization & Helpers
# ============================================================

def normalize_text(value) -> str:
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def yes_no_label(value) -> str | None:
    norm = normalize_text(value)
    if not norm:
        return None
    first = norm.split()[0]
    if first in ("yes", "true"):
        return "yes"
    if first in ("no", "false"):
        return "no"
    return None


@dataclass
class DiagnosticContextItem:
    source: str


@dataclass
class DiagnosticContext:
    items: list[DiagnosticContextItem]


# ============================================================
# Candidate Verifiers (No Gold Leakage)
# ============================================================

class AnswerTypeGuard:
    """
    Detects publisher/source-as-answer collision when the question
    asks for an entity (organization, company, person, university, country)
    other than a news source/publisher.
    """

    NON_PUBLISHER_ENTITY_STARTERS = [
        "which organization",
        "which company",
        "which country",
        "who is the individual",
        "who is",
        "which person",
        "which university",
        "what company",
        "what organization",
    ]

    PUBLISHER_ENTITY_STARTERS = [
        "which news source",
        "which publisher",
        "which article",
        "which outlet",
        "which publication",
        "which blog",
    ]

    @classmethod
    def check(
        cls,
        *,
        question: str,
        draft_direct_answer: str | None,
        context_sources: list[str],
    ) -> tuple[bool, str]:
        if not draft_direct_answer:
            return False, "No draft answer"

        q_norm = (question or "").lower()

        # If question explicitly asks for a news source/publisher, source names are legitimate answers
        if any(p in q_norm for p in cls.PUBLISHER_ENTITY_STARTERS):
            return False, "Question asks for a news source/publisher"

        # Check if question is an entity question asking for a non-publisher entity
        is_entity_q = any(q_norm.startswith(p) or (f" {p} " in q_norm) for p in cls.NON_PUBLISHER_ENTITY_STARTERS)
        if not is_entity_q:
            return False, "Not an entity question"

        norm_answer = normalize_text(draft_direct_answer)
        if not norm_answer:
            return False, "Empty answer"

        for source in context_sources:
            if not source:
                continue
            norm_source = normalize_text(source)
            primary_source = normalize_text(source.split("|")[0])

            if norm_answer == norm_source or (primary_source and norm_answer == primary_source):
                return True, f"Publisher collision: answer '{draft_direct_answer}' matches source '{source}'"

        return False, "No collision"


class RequirementCoverageGuard:
    """
    Checks whether all explicitly required sources for multi-source questions
    are covered by the surviving grounded claims.
    If coverage is incomplete for a multi-source question, any conclusion
    cannot be safely affirmed -> returns UNKNOWN / abstains.
    """

    def __init__(self):
        self.source_guard = ExplicitSourceCoverageGuard()

    def check(
        self,
        *,
        question: str,
        cited_sources: list[str],
    ) -> tuple[bool, str, list[str], list[str]]:
        context = DiagnosticContext(
            items=[DiagnosticContextItem(source=s) for s in cited_sources]
        )
        res = self.source_guard.check(query=question, context=context)

        # Multi-source question with missing required sources
        if len(res.required_sources) >= 2 and not res.satisfied:
            return False, f"Missing required sources: {res.missing_sources}", res.required_sources, res.covered_sources

        return True, "Coverage satisfied or not multi-source", res.required_sources, res.covered_sources


class StructuredConclusionVerifier:
    """
    Stage A: Requirement Coverage Verification
    Stage B: Structured Semantic Conclusion
    """

    def __init__(self):
        self.type_guard = AnswerTypeGuard()
        self.coverage_guard = RequirementCoverageGuard()
        self.relation_resolver = RelationAwareAnswerResolver()

    def verify(
        self,
        *,
        question: str,
        draft_direct_answer: str | None,
        grounded_facts: list[str],
        cited_sources: list[str],
    ) -> dict:
        # Step 1: Check AnswerTypeGuard (Publisher-as-answer collision)
        has_collision, collision_reason = self.type_guard.check(
            question=question,
            draft_direct_answer=draft_direct_answer,
            context_sources=cited_sources,
        )

        if has_collision:
            return {
                "final_direct_answer": "UNKNOWN",
                "status": "COLLISION_REJECTED",
                "applied_mechanism": "AnswerTypeGuard",
                "reason": collision_reason,
            }

        # Step 2: Check Existing RelationAwareAnswerResolver (Exact Predicate Consistency)
        if grounded_facts and len(grounded_facts) >= 2:
            rel_res = self.relation_resolver.resolve(
                query=question,
                facts=grounded_facts,
            )
            if rel_res.applied and rel_res.resolved_answer:
                return {
                    "final_direct_answer": rel_res.resolved_answer,
                    "status": "RELATION_RESOLVED",
                    "applied_mechanism": "RelationAwareAnswerResolver",
                    "reason": rel_res.reason,
                }

        # Step 3: Check RequirementCoverageGuard
        cov_ok, cov_reason, req_srcs, cov_srcs = self.coverage_guard.check(
            question=question,
            cited_sources=cited_sources,
        )

        if not cov_ok:
            # Multi-source question is undercovered
            return {
                "final_direct_answer": "UNKNOWN",
                "status": "UNDERCOVERED_ABSTAIN",
                "applied_mechanism": "RequirementCoverageGuard",
                "reason": cov_reason,
            }

        # Step 4: Default pass-through of draft answer
        return {
            "final_direct_answer": draft_direct_answer,
            "status": "PRESERVED",
            "applied_mechanism": "None",
            "reason": "Draft answer preserved",
        }


# ============================================================
# Ablation Runner
# ============================================================

def run_ablation():
    records = load_records()
    doc_catalog = build_document_source_catalog()

    verifier = StructuredConclusionVerifier()

    # Metrics containers
    configurations = {
        "baseline": {},
        "answer_type_guard_only": {},
        "requirement_coverage_guard_only": {},
        "relation_resolver_only": {},
        "full_structured_conclusion_verifier": {},
    }

    results_by_config = {cfg: [] for cfg in configurations}

    for record in records:
        rec_id = record.get("id")
        q = record.get("question", "")
        q_type = record.get("question_type", "")
        gold_ans = record.get("gold_answer")
        draft_ans = record.get("draft_direct_answer")
        prod_direct_ans = record.get("direct_answer")
        abstained = bool(record.get("abstained", False))
        smoke_correct = record.get("smoke_answer_correct")

        # Extract cited sources
        cited_doc_ids = record.get("cited_document_ids", []) or []
        cited_sources = []
        seen = set()
        for d in cited_doc_ids:
            s = doc_catalog.get(str(d), {}).get("source", "")
            if s and s not in seen:
                seen.add(s)
                cited_sources.append(s)

        # Grounded facts from raw_answer or answer bullets
        raw_ans = record.get("raw_answer", "") or ""
        facts = []
        for line in raw_ans.splitlines():
            line = line.strip()
            if line.startswith("- "):
                facts.append(line[2:].strip())

        gold_yn = yes_no_label(gold_ans)

        # Evaluate each configuration
        # 1. Baseline
        baseline_ans = prod_direct_ans

        # 2. AnswerTypeGuard only
        col, _ = AnswerTypeGuard.check(question=q, draft_direct_answer=draft_ans, context_sources=cited_sources)
        type_guard_ans = "UNKNOWN" if col else draft_ans

        # 3. RequirementCoverageGuard only
        cov_ok, _, _, _ = RequirementCoverageGuard().check(question=q, cited_sources=cited_sources)
        cov_guard_ans = "UNKNOWN" if (not cov_ok and not abstained) else draft_ans

        # 4. RelationResolver only
        if facts and len(facts) >= 2:
            rel_res = RelationAwareAnswerResolver().resolve(query=q, facts=facts)
            rel_only_ans = rel_res.resolved_answer if (rel_res.applied and rel_res.resolved_answer) else draft_ans
        else:
            rel_only_ans = draft_ans

        # 5. Full StructuredConclusionVerifier
        if abstained:
            full_res = {"final_direct_answer": None, "status": "ABSTAINED", "applied_mechanism": "None"}
        else:
            full_res = verifier.verify(
                question=q,
                draft_direct_answer=draft_ans,
                grounded_facts=facts,
                cited_sources=cited_sources,
            )

        config_outputs = {
            "baseline": baseline_ans,
            "answer_type_guard_only": type_guard_ans,
            "requirement_coverage_guard_only": cov_guard_ans,
            "relation_resolver_only": rel_only_ans,
            "full_structured_conclusion_verifier": full_res["final_direct_answer"],
        }

        for cfg, ans in config_outputs.items():
            results_by_config[cfg].append({
                "id": rec_id,
                "question": q,
                "gold_answer": gold_ans,
                "predicted_answer": ans,
                "abstained": abstained or ans in (None, "UNKNOWN", "INSUFFICIENT_EVIDENCE"),
                "smoke_correct": smoke_correct if cfg == "baseline" else None,
            })

    # Compute evaluation metrics for each config
    summary_by_config = {}

    for cfg, recs in results_by_config.items():
        # Evaluate on the 9 answered answerable cases from Smoke20
        answered_baseline = [r for r in records if r["question_type"] != "null_query" and not r.get("abstained", False)]

        wrong_to_right = 0
        wrong_to_unknown = 0
        right_to_wrong = 0
        right_to_unknown = 0
        unchanged_correct = 0
        unchanged_wrong = 0

        for r_base, r_cfg in zip(records, recs):
            if r_base["question_type"] == "null_query" or r_base.get("abstained", False):
                continue

            gold = r_base.get("gold_answer")
            gold_yn = yes_no_label(gold)
            draft = r_base.get("draft_direct_answer")
            draft_yn = yes_no_label(draft)
            base_pred = r_base.get("direct_answer")
            base_yn = yes_no_label(base_pred)
            base_correct = r_base.get("smoke_answer_correct", False)

            cfg_pred = r_cfg["predicted_answer"]
            cfg_yn = yes_no_label(cfg_pred)
            cfg_is_unknown = cfg_pred in (None, "UNKNOWN", "INSUFFICIENT_EVIDENCE")

            # Evaluate correctness of cfg_pred
            if cfg_is_unknown:
                cfg_correct = None
            elif gold_yn is not None:
                cfg_correct = (cfg_yn == gold_yn)
            else:
                cfg_correct = (normalize_text(cfg_pred) == normalize_text(gold))

            # Transitions relative to base_correct
            if base_correct:
                if cfg_correct is True:
                    unchanged_correct += 1
                elif cfg_is_unknown:
                    right_to_unknown += 1
                else:
                    right_to_wrong += 1
            else:
                # Baseline was wrong
                if cfg_correct is True:
                    wrong_to_right += 1
                elif cfg_is_unknown:
                    wrong_to_unknown += 1
                else:
                    unchanged_wrong += 1

        summary_by_config[cfg] = {
            "total_answered": len(answered_baseline),
            "wrong_to_right": wrong_to_right,
            "wrong_to_unknown": wrong_to_unknown,
            "right_to_wrong": right_to_wrong,
            "right_to_unknown": right_to_unknown,
            "unchanged_correct": unchanged_correct,
            "unchanged_wrong": unchanged_wrong,
            "false_positive_count": unchanged_wrong + right_to_wrong,
            "false_positive_rate": (unchanged_wrong + right_to_wrong) / len(answered_baseline),
        }

    return {
        "summary": summary_by_config,
        "detail": results_by_config,
    }


def main():
    print("=" * 80)
    print("RUNNING SEMANTIC CONCLUSION AND GROUNDING ABLATION")
    print("=" * 80)

    ablation_res = run_ablation()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(ablation_res, f, indent=2, ensure_ascii=False)

    print("\nABLATION SUMMARY:")
    print(json.dumps(ablation_res["summary"], indent=2))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

