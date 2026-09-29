import ast

from typing import Any

from adaptive_agentic_rag.agents.query_router import (
    QueryRouter,
)

from adaptive_agentic_rag.agents.evidence_grader import (
    EvidenceGrader,
)

from adaptive_agentic_rag.agents.query_rewriter import (
    QueryRewriter,
)

from adaptive_agentic_rag.agents.answer_grader import (
    AnswerGrader,
)

from adaptive_agentic_rag.retrieval.adaptive_retriever import (
    AdaptiveRetriever,
)

from adaptive_agentic_rag.generation.context_builder import (
    ContextBuilder,
)

from adaptive_agentic_rag.generation.generator import (
    GroundedGenerator,
)

from adaptive_agentic_rag.orchestration.state import (
    AgentState,
)

from adaptive_agentic_rag.orchestration.constrained_semantic_rescue import (
    ConstrainedSemanticRescue,
)

from adaptive_agentic_rag.orchestration.explicit_source_coverage import (
    ExplicitSourceCoverageGuard,
)


class RAGNodes:
    """
    LangGraph node implementations.

    Shared models/services live here.
    Per-request information lives in AgentState.

    Evidence architecture
    ---------------------

        EvidenceGrader V2
                ↓
        Explicit Source Coverage
                ↓
        ┌─────────────────────────────┐
        │                             │
        │ source missing              │ sources present
        ↓                             ↓
    hard rejection              V2 sufficient?
                                      │
                           ┌──────────┴──────────┐
                           │                     │
                          yes                    no
                           ↓                     ↓
                     V2 fast path       Constrained Semantic
                                              Rescue

    Retry architecture
    ------------------

        explicit source miss
                ↓
        AdaptiveRetryPolicy
                ↓
        CorpusSourceAvailability
                ↓
        rewrite_query
                ↓
        retry_target_sources
                ↓
        source-targeted retrieval
    """

    def __init__(
        self,
    ):

        # ==================================================
        # Core services
        # ==================================================

        self.router = (
            QueryRouter()
        )


        self.retriever = (
            AdaptiveRetriever()
        )


        self.context_builder = (
            ContextBuilder()
        )


        self.evidence_grader = (
            EvidenceGrader()
        )


        self.query_rewriter = (
            QueryRewriter()
        )


        # ==================================================
        # Hard structural source coverage
        # ==================================================

        self.source_coverage_guard = (
            ExplicitSourceCoverageGuard()
        )


        # ==================================================
        # Semantic rescue
        #
        # Reuse already-loaded reranker.
        # ==================================================

        self.semantic_rescue = (
            ConstrainedSemanticRescue(
                reranker=(
                    self.retriever
                    .reranked
                    .reranker
                ),

                evidence_grader=(
                    self.evidence_grader
                ),
            )
        )


        # ==================================================
        # Generator
        #
        # Reuse same BGE reranker.
        # ==================================================

        self.generator = (
            GroundedGenerator(
                reranker=(
                    self.retriever
                    .reranked
                    .reranker
                )
            )
        )


        # ==================================================
        # Answer grader
        #
        # Reuse dense embedder.
        # ==================================================

        self.answer_grader = (
            AnswerGrader(
                embedder=(
                    self.retriever
                    .dense
                    .embedder
                )
            )
        )


    # ======================================================
    # Utility
    # ======================================================

    @staticmethod
    def _safe_list(
        value: Any,
    ) -> list[Any]:

        if value is None:

            return []


        if isinstance(
            value,
            list,
        ):

            return value


        return list(
            value
        )


    # ======================================================
    # Evidence telemetry list extraction
    #
    # QueryRewriter V2 and source-targeted retry consume
    # structural source telemetry produced by
    # ExplicitSourceCoverageGuard.
    #
    # Malformed/missing telemetry fails closed to [].
    # ======================================================

    @staticmethod
    def _extract_reason_list(
        reasons: list[str],
        prefix: str,
    ) -> list[str]:

        for reason in (
            reasons
            or []
        ):

            if not isinstance(
                reason,
                str,
            ):

                continue


            if not reason.startswith(
                prefix
            ):

                continue


            raw_value = (
                reason[
                    len(
                        prefix
                    ):
                ]
                .strip()
            )


            try:

                parsed = (
                    ast.literal_eval(
                        raw_value
                    )
                )


            except (
                ValueError,
                SyntaxError,
            ):

                return []


            if not isinstance(
                parsed,
                list,
            ):

                return []


            return [
                str(
                    value
                )

                for value
                in parsed
            ]


        return []


    # ======================================================
    # Structured-result accessor
    # ======================================================

    @staticmethod
    def _result_value(
        result,
        name: str,
        default=None,
    ):

        if result is None:

            return default


        if isinstance(
            result,
            dict,
        ):

            return result.get(
                name,
                default,
            )


        return getattr(
            result,
            name,
            default,
        )


    # ======================================================
    # Citation validity extraction
    # ======================================================

    @staticmethod
    def _extract_citation_valid(
        generation_result,
    ) -> bool | None:

        direct_value = getattr(
            generation_result,
            "citation_valid",
            None,
        )


        if direct_value is not None:

            return bool(
                direct_value
            )


        nested_names = [
            "citation_validation",
            "citation_result",
            "citation_report",
        ]


        validity_names = [
            "valid",
            "is_valid",
            "citation_valid",
        ]


        for nested_name in nested_names:

            nested = getattr(
                generation_result,
                nested_name,
                None,
            )


            if nested is None:

                continue


            for validity_name in (
                validity_names
            ):

                value = getattr(
                    nested,
                    validity_name,
                    None,
                )


                if value is not None:

                    return bool(
                        value
                    )


        return None


    # ======================================================
    # GenerationResult → state update
    # ======================================================

    def _generation_state_update(
        self,
        result,
    ) -> dict:

        final_answer = getattr(
            result,
            "answer",
            None,
        )


        if final_answer is None:

            final_answer = getattr(
                result,
                "final_answer",
                None,
            )


        return {
            "generation_result":
                result,

            "generation_model_name":
                getattr(
                    result,
                    "model_name",
                    None,
                ),

            "raw_answer":
                getattr(
                    result,
                    "raw_answer",
                    None,
                ),

            "final_answer":
                final_answer,

            "abstained":
                bool(
                    getattr(
                        result,
                        "abstained",
                        False,
                    )
                ),

            "supported_claims":
                int(
                    getattr(
                        result,
                        "supported_claims",
                        0,
                    )
                ),

            "unsupported_claims":
                int(
                    getattr(
                        result,
                        "unsupported_claims",
                        0,
                    )
                ),

            "relevant_claims":
                int(
                    getattr(
                        result,
                        "relevant_claims",
                        0,
                    )
                ),

            "filtered_irrelevant_claims":
                int(
                    getattr(
                        result,
                        "filtered_irrelevant_claims",
                        0,
                    )
                ),

            "citation_valid":
                self._extract_citation_valid(
                    result
                ),
        }


    # ======================================================
    # Semantic rescue telemetry
    # ======================================================

    def _semantic_rescue_reasons(
        self,
        rescue_result,
    ) -> list[str]:

        sufficient = bool(
            self._result_value(
                rescue_result,
                "sufficient",
                False,
            )
        )


        semantic_status = (
            "sufficient"
            if sufficient
            else "insufficient"
        )


        reasons = [
            "semantic_rescue_attempted=true",

            (
                "semantic_rescue_sufficient="
                f"{str(sufficient).lower()}"
            ),

            (
                f"semantic_rescue="
                f"{semantic_status}"
            ),
        ]


        threshold = (
            self._result_value(
                rescue_result,
                "threshold",
                None,
            )
        )


        if threshold is not None:

            reasons.append(
                (
                    "semantic_rescue_threshold="
                    f"{threshold}"
                )
            )


        required_fraction = (
            self._result_value(
                rescue_result,
                "required_fraction",
                None,
            )
        )


        if required_fraction is not None:

            reasons.append(
                (
                    "semantic_rescue_required_fraction="
                    f"{required_fraction}"
                )
            )


        requirement_count = (
            self._result_value(
                rescue_result,
                "requirement_count",
                None,
            )
        )


        supported_count = (
            self._result_value(
                rescue_result,
                "supported_requirement_count",
                None,
            )
        )


        required_count = (
            self._result_value(
                rescue_result,
                "required_requirement_count",
                None,
            )
        )


        if requirement_count is not None:

            reasons.append(
                (
                    "semantic_rescue_requirement_count="
                    f"{requirement_count}"
                )
            )


        if supported_count is not None:

            reasons.append(
                (
                    "semantic_rescue_supported_requirements="
                    f"{supported_count}"
                )
            )


        if required_count is not None:

            reasons.append(
                (
                    "semantic_rescue_required_requirements="
                    f"{required_count}"
                )
            )


        missing_query_sources = (
            self._result_value(
                rescue_result,
                "missing_query_sources",
                None,
            )
        )


        if missing_query_sources is not None:

            reasons.append(
                (
                    "semantic_rescue_missing_query_sources="
                    f"{list(missing_query_sources)}"
                )
            )


        supporting_document_ids = (
            self._result_value(
                rescue_result,
                "supporting_document_ids",
                None,
            )
        )


        if supporting_document_ids is not None:

            reasons.append(
                (
                    "semantic_rescue_supporting_document_ids="
                    f"{list(supporting_document_ids)}"
                )
            )


        if sufficient:

            reasons.append(
                (
                    "evidence_path="
                    "constrained_semantic_rescue"
                )
            )


        else:

            reasons.append(
                (
                    "evidence_path="
                    "constrained_semantic_rescue_reject"
                )
            )


        return reasons


    # ======================================================
    # Node 1
    # Query routing
    # ======================================================

    def route_query(
        self,
        state: AgentState,
    ) -> dict:

        query = (
            state[
                "current_query"
            ]
        )


        decision = (
            self.router.route(
                query
            )
        )


        return {
            "query_type":
                decision[
                    "query_type"
                ],

            "retrieval_strategy":
                decision[
                    "retrieval_strategy"
                ],

            "use_reranker":
                decision[
                    "rerank"
                ],

            "use_mmr":
                decision[
                    "mmr"
                ],
        }


    # ======================================================
    # Node 2
    # Retrieval
    #
    # First attempt:
    #
    #     retry_target_sources=[]
    #     -> canonical retrieval unchanged
    #
    # Approved retry:
    #
    #     retry_target_sources=["The Age"]
    #     -> source-targeted candidate injection
    # ======================================================

    def retrieve(
        self,
        state: AgentState,
    ) -> dict:

        query = (
            state[
                "current_query"
            ]
        )


        retry_target_sources = list(
            state.get(
                "retry_target_sources",
                [],
            )
            or []
        )


        retrieval_output = (
            self.retriever.search(
                query,
                top_k=
                    10,

                target_sources=
                    retry_target_sources,
            )
        )


        results = (
            retrieval_output[
                "results"
            ]
        )


        return {
            "retrieved_results":
                results,
        }


    # ======================================================
    # Node 3
    # Context construction
    #
    # ContextBuilder always uses original user semantics.
    # ======================================================

    def build_context(
        self,
        state: AgentState,
    ) -> dict:

        results = (
            state[
                "retrieved_results"
            ]
        )


        context_query = (
            state.get(
                "original_query"
            )
            or
            state.get(
                "current_query",
                "",
            )
        )


        context = (
            self.context_builder.build(
                results,

                query=
                    context_query,
            )
        )


        return {
            "context":
                context,
        }


    # ======================================================
    # Node 4
    # Evidence grading
    # ======================================================

    def grade_evidence(
        self,
        state: AgentState,
    ) -> dict:

        original_query = (
            state[
                "original_query"
            ]
        )


        context = (
            state[
                "context"
            ]
        )


        query_type = (
            state[
                "query_type"
            ]
        )


        if context is None:

            raise ValueError(
                "Cannot grade evidence "
                "without a built context."
            )


        if query_type is None:

            raise ValueError(
                "Cannot grade evidence "
                "without query_type."
            )


        # ==================================================
        # Stage A
        # Base EvidenceGrader V2
        # ==================================================

        grade = (
            self.evidence_grader.grade(
                query=
                    original_query,

                context=
                    context,

                query_type=
                    query_type,
            )
        )


        base_reasons = list(
            getattr(
                grade,
                "reasons",
                [],
            )
            or []
        )


        # ==================================================
        # Stage B
        # Explicit Source Coverage
        # ==================================================

        source_coverage_guard = (
            getattr(
                self,
                "source_coverage_guard",
                None,
            )
        )


        if source_coverage_guard is None:

            source_coverage_guard = (
                ExplicitSourceCoverageGuard()
            )


            self.source_coverage_guard = (
                source_coverage_guard
            )


        source_coverage = (
            source_coverage_guard.check(
                query=
                    original_query,

                context=
                    context,
            )
        )


        # ==================================================
        # HARD structural source rejection
        # ==================================================

        if not (
            source_coverage.satisfied
        ):

            reasons = list(
                base_reasons
            )


            reasons.extend(
                [
                    (
                        "Explicit source coverage failed."
                    ),

                    (
                        "required_sources="
                        f"{source_coverage.required_sources}"
                    ),

                    (
                        "available_sources="
                        f"{source_coverage.available_sources}"
                    ),

                    (
                        "covered_sources="
                        f"{source_coverage.covered_sources}"
                    ),

                    (
                        "missing_sources="
                        f"{source_coverage.missing_sources}"
                    ),

                    (
                        "evidence_path="
                        "explicit_source_coverage_reject"
                    ),
                ]
            )


            return {
                "evidence_sufficient":
                    False,

                "evidence_score":
                    grade.evidence_score,

                "evidence_reasons":
                    reasons,
            }


        # ==================================================
        # Stage C
        # V2 fast path
        # ==================================================

        if grade.sufficient:

            reasons = list(
                base_reasons
            )


            reasons.extend(
                [
                    (
                        "explicit_source_coverage="
                        "satisfied"
                    ),

                    (
                        "required_sources="
                        f"{source_coverage.required_sources}"
                    ),

                    (
                        "covered_sources="
                        f"{source_coverage.covered_sources}"
                    ),

                    "evidence_path=v2",
                ]
            )


            return {
                "evidence_sufficient":
                    True,

                "evidence_score":
                    grade.evidence_score,

                "evidence_reasons":
                    reasons,
            }


        # ==================================================
        # Stage D
        # Constrained Semantic Rescue
        # ==================================================

        rescue_result = (
            self.semantic_rescue.analyze(
                query=
                    original_query,

                context=
                    context,

                query_type=
                    query_type,
            )
        )


        rescue_sufficient = bool(
            self._result_value(
                rescue_result,
                "sufficient",
                False,
            )
        )


        reasons = list(
            base_reasons
        )


        reasons.extend(
            [
                (
                    "explicit_source_coverage="
                    "satisfied"
                ),

                (
                    "required_sources="
                    f"{source_coverage.required_sources}"
                ),

                (
                    "covered_sources="
                    f"{source_coverage.covered_sources}"
                ),
            ]
        )


        reasons.extend(
            self._semantic_rescue_reasons(
                rescue_result
            )
        )


        return {
            "evidence_sufficient":
                rescue_sufficient,

            "evidence_score":
                grade.evidence_score,

            "evidence_reasons":
                reasons,
        }


    # ======================================================
    # Node 5
    # Query rewriting
    #
    # Critical handoff:
    #
    # evidence_reasons
    #      ↓
    # missing_sources
    #      ├── QueryRewriter V2
    #      └── retry_target_sources
    #
    # retry_target_sources survives evidence reset and is
    # consumed by the next retrieve node.
    # ======================================================

    def rewrite_query(
        self,
        state: AgentState,
    ) -> dict:

        original_query = (
            state[
                "original_query"
            ]
        )


        query_type = (
            state[
                "query_type"
            ]
        )


        if query_type is None:

            raise ValueError(
                "Cannot rewrite query "
                "without query_type."
            )


        attempt = (
            state[
                "retry_count"
            ]
            +
            1
        )


        # ==================================================
        # Capture telemetry BEFORE clearing evidence state.
        # ==================================================

        evidence_reasons = list(
            state.get(
                "evidence_reasons",
                [],
            )
            or []
        )


        required_sources = (
            self._extract_reason_list(
                evidence_reasons,
                "required_sources=",
            )
        )


        covered_sources = (
            self._extract_reason_list(
                evidence_reasons,
                "covered_sources=",
            )
        )


        missing_sources = (
            self._extract_reason_list(
                evidence_reasons,
                "missing_sources=",
            )
        )


        rewritten_query = (
            self.query_rewriter.rewrite(
                query=
                    original_query,

                query_type=
                    query_type,

                attempt=
                    attempt,

                required_sources=
                    required_sources,

                covered_sources=
                    covered_sources,

                missing_sources=
                    missing_sources,
            )
        )


        return {
            "current_query":
                rewritten_query,

            "retry_count":
                attempt,

            "rewritten":
                True,

            # ==============================================
            # CRITICAL:
            #
            # This was missing in the actual production
            # nodes.py and was why source-targeted retrieval
            # never activated during frozen500.
            # ==============================================

            "retry_target_sources":
                list(
                    missing_sources
                ),

            # ----------------------------------------------
            # Reset previous retrieval round.
            # ----------------------------------------------

            "retrieved_results":
                [],

            "context":
                None,

            "evidence_sufficient":
                None,

            "evidence_score":
                None,

            "evidence_reasons":
                [],
        }


    # ======================================================
    # Node 6
    # Grounded generation
    # ======================================================

    def generate(
        self,
        state: AgentState,
    ) -> dict:

        original_query = (
            state[
                "original_query"
            ]
        )


        context = (
            state[
                "context"
            ]
        )


        evidence_sufficient = (
            state[
                "evidence_sufficient"
            ]
        )


        if context is None:

            raise ValueError(
                "Cannot generate answer "
                "without context."
            )


        if evidence_sufficient is not True:

            raise ValueError(
                "Generate node should only "
                "run when evidence is sufficient."
            )


        result = (
            self.generator.generate(
                query=
                    original_query,

                context=
                    context,

                evidence_sufficient=
                    True,
            )
        )


        return (
            self._generation_state_update(
                result
            )
        )


    # ======================================================
    # Node 7
    # Explicit abstention
    # ======================================================

    def abstain(
        self,
        state: AgentState,
    ) -> dict:

        original_query = (
            state[
                "original_query"
            ]
        )


        context = (
            state[
                "context"
            ]
        )


        if context is None:

            raise ValueError(
                "Cannot create abstention "
                "without context."
            )


        result = (
            self.generator.generate(
                query=
                    original_query,

                context=
                    context,

                evidence_sufficient=
                    False,
            )
        )


        return (
            self._generation_state_update(
                result
            )
        )


    # ======================================================
    # Node 8
    # Final answer grading
    # ======================================================

    def grade_answer(
        self,
        state: AgentState,
    ) -> dict:

        generation_result = (
            state[
                "generation_result"
            ]
        )


        if generation_result is None:

            raise ValueError(
                "Cannot grade answer "
                "without GenerationResult."
            )


        original_query = (
            state[
                "original_query"
            ]
        )


        evidence_sufficient = (
            state[
                "evidence_sufficient"
            ]
        )


        if evidence_sufficient is None:

            raise ValueError(
                "Cannot grade answer "
                "before evidence grading."
            )


        grade = (
            self.answer_grader.grade(
                query=
                    original_query,

                generation_result=
                    generation_result,

                evidence_sufficient=
                    evidence_sufficient,
            )
        )


        return {
            "answer_grade":
                grade,

            "answer_passed":
                getattr(
                    grade,
                    "passed",
                    None,
                ),

            "answer_relevance_score":
                getattr(
                    grade,
                    "relevance_score",
                    None,
                ),

            "answer_grade_reasons":
                self._safe_list(
                    getattr(
                        grade,
                        "reasons",
                        [],
                    )
                ),
        }


    # ======================================================
    # Cleanup
    # ======================================================

    def close(
        self,
    ):

        close_method = getattr(
            self.retriever,
            "close",
            None,
        )


        if callable(
            close_method
        ):

            close_method()