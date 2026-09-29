import re
from dataclasses import dataclass

from adaptive_agentic_rag.generation.relation_aware_answer_resolver import (
    RelationAwareAnswerResolver,
    RelationResolution,
)


@dataclass
class StructuredVerificationResult:
    applied: bool
    resolved_answer: str | None
    status: str
    applied_mechanism: str
    reason: str
    required_sources: list[str]
    covered_sources: list[str]
    missing_sources: list[str]
    relation_resolution: RelationResolution | None = None


# ============================================================
# Stage B Helper: AnswerTypeGuard
# ============================================================

class AnswerTypeGuard:
    """
    Detects publisher/source-as-answer collision when the question
    asks for an entity (organization, company, person, country, etc.)
    distinct from the publisher or news source.
    """

    NON_PUBLISHER_ENTITY_STARTERS = [
        "which organization",
        "which company",
        "which country",
        "who is the individual",
        "who is",
        "which person",
        "which player",
        "which team",
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
        "what news source",
        "what publisher",
        "what publication",
    ]

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))

    @classmethod
    def check(
        cls,
        *,
        question: str,
        draft_direct_answer: str | None,
        context_sources: list[str],
    ) -> tuple[bool, str]:
        if not draft_direct_answer:
            return False, "No draft direct answer."

        q_norm = (question or "").lower()

        # If question explicitly asks for a news source/publisher, source names are legitimate answers
        if any(p in q_norm for p in cls.PUBLISHER_ENTITY_STARTERS):
            return False, "Question explicitly requests a news source/publisher entity."

        # Check if question is an entity question asking for a non-publisher entity
        is_non_publisher_entity_q = any(
            q_norm.startswith(p) or (f" {p} " in q_norm)
            for p in cls.NON_PUBLISHER_ENTITY_STARTERS
        )
        if not is_non_publisher_entity_q:
            return False, "Question is not a non-publisher entity query."

        norm_answer = cls._normalize(draft_direct_answer)
        if not norm_answer:
            return False, "Normalized draft answer is empty."

        for source in context_sources:
            if not source:
                continue
            norm_source = cls._normalize(source)
            primary_source = cls._normalize(source.split("|")[0])

            if norm_answer == norm_source or (primary_source and norm_answer == primary_source):
                return True, (
                    f"Publisher-as-answer collision: draft answer '{draft_direct_answer}' "
                    f"matches context source '{source}' for a non-publisher entity question."
                )

        return False, "No publisher-as-answer collision detected."


# ============================================================
# Stage A: RequirementCoverageGuard
# ============================================================

@dataclass
class _DiagnosticContextItem:
    source: str


@dataclass
class _DiagnosticContext:
    items: list[_DiagnosticContextItem]


class RequirementCoverageGuard:
    """
    Verifies whether all explicitly required sources for multi-source questions
    are covered by the surviving grounded claims.
    """

    def __init__(self, source_guard=None):
        self._source_guard = source_guard

    def _get_source_guard(self):
        if self._source_guard is None:
            from adaptive_agentic_rag.orchestration.explicit_source_coverage import (
                ExplicitSourceCoverageGuard,
            )
            self._source_guard = ExplicitSourceCoverageGuard()
        return self._source_guard

    def check(
        self,
        *,
        query: str | None = None,
        question: str | None = None,
        covered_sources: list[str],
    ) -> tuple[bool, str, list[str], list[str], list[str]]:
        q_text = query if query is not None else (question or "")
        guard = self._get_source_guard()
        context = _DiagnosticContext(
            items=[_DiagnosticContextItem(source=s) for s in covered_sources]
        )
        res = guard.check(query=q_text, context=context)

        # Multi-source question with missing required sources
        if len(res.required_sources) >= 2 and not res.satisfied:
            return (
                False,
                f"Multi-source question requires {len(res.required_sources)} sources ({res.required_sources}), "
                f"but only {len(res.covered_sources)} are covered ({res.covered_sources}). Missing: {res.missing_sources}.",
                res.required_sources,
                res.covered_sources,
                res.missing_sources,
            )

        return (
            True,
            "Requirement coverage satisfied or single-source query.",
            res.required_sources,
            res.covered_sources,
            res.missing_sources,
        )


# ============================================================
# Orchestrated Verifier: StructuredConclusionVerifier
# ============================================================

class StructuredConclusionVerifier:
    """
    Decomposed two-stage semantic verification:

        Grounded Relevant Claims
                   ↓
        Stage A: Grounded Requirement Coverage (RequirementCoverageGuard)
                   ↓
        Stage B: Structured Semantic Conclusion
                 ├── AnswerTypeGuard (Publisher-as-answer collision detection)
                 ├── RelationAwareAnswerResolver (Exact consistency resolution)
                 └── RequirementCoverage Enforcement (Abstain/Unknown on undercoverage)
                   ↓
        Final Direct Answer
    """

    def __init__(
        self,
        *,
        relation_resolver: RelationAwareAnswerResolver | None = None,
        coverage_guard: RequirementCoverageGuard | None = None,
    ):
        self.type_guard = AnswerTypeGuard()
        self.coverage_guard = coverage_guard or RequirementCoverageGuard()
        self.relation_resolver = (
            relation_resolver or RelationAwareAnswerResolver()
        )

    def verify(
        self,
        *,
        query: str,
        draft_direct_answer: str | None,
        grounded_facts: list[str],
        covered_sources: list[str],
        all_context_sources: list[str] | None = None,
    ) -> StructuredVerificationResult:
        context_sources = (
            all_context_sources
            if all_context_sources is not None
            else covered_sources
        )

        # ----------------------------------------------------
        # Step 1: Stage B AnswerTypeGuard
        # ----------------------------------------------------
        has_collision, collision_reason = self.type_guard.check(
            question=query,
            draft_direct_answer=draft_direct_answer,
            context_sources=context_sources,
        )

        if has_collision:
            cov_ok, _, req_s, cov_s, miss_s = self.coverage_guard.check(
                query=query,
                covered_sources=covered_sources,
            )
            return StructuredVerificationResult(
                applied=True,
                resolved_answer="UNKNOWN",
                status="COLLISION_REJECTED",
                applied_mechanism="AnswerTypeGuard",
                reason=collision_reason,
                required_sources=req_s,
                covered_sources=cov_s,
                missing_sources=miss_s,
                relation_resolution=None,
            )

        # ----------------------------------------------------
        # Step 2: Stage B Existing Exact Consistency Resolver
        # ----------------------------------------------------
        relation_res = None
        if grounded_facts and len(grounded_facts) >= 2:
            relation_res = self.relation_resolver.resolve(
                query=query,
                facts=grounded_facts,
            )
            if relation_res.applied and relation_res.resolved_answer:
                cov_ok, _, req_s, cov_s, miss_s = self.coverage_guard.check(
                    query=query,
                    covered_sources=covered_sources,
                )
                return StructuredVerificationResult(
                    applied=True,
                    resolved_answer=relation_res.resolved_answer,
                    status="RELATION_RESOLVED",
                    applied_mechanism="RelationAwareAnswerResolver",
                    reason=relation_res.reason,
                    required_sources=req_s,
                    covered_sources=cov_s,
                    missing_sources=miss_s,
                    relation_resolution=relation_res,
                )

        # ----------------------------------------------------
        # Step 3: Stage A Requirement Coverage Verification
        # ----------------------------------------------------
        cov_ok, cov_reason, req_s, cov_s, miss_s = self.coverage_guard.check(
            query=query,
            covered_sources=covered_sources,
        )

        if not cov_ok:
            # Undercovered multi-source query
            return StructuredVerificationResult(
                applied=True,
                resolved_answer="UNKNOWN",
                status="UNDERCOVERED_ABSTAIN",
                applied_mechanism="RequirementCoverageGuard",
                reason=cov_reason,
                required_sources=req_s,
                covered_sources=cov_s,
                missing_sources=miss_s,
                relation_resolution=relation_res,
            )

        # ----------------------------------------------------
        # Step 4: Default Preservation
        # ----------------------------------------------------
        return StructuredVerificationResult(
            applied=False,
            resolved_answer=draft_direct_answer,
            status="PRESERVED",
            applied_mechanism="None",
            reason="Draft direct answer preserved under complete requirement coverage and valid answer typing.",
            required_sources=req_s,
            covered_sources=cov_s,
            missing_sources=miss_s,
            relation_resolution=relation_res,
        )
