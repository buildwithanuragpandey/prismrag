from langgraph.graph import (
    START,
    END,
    StateGraph,
)

from adaptive_agentic_rag.orchestration.state import (
    AgentState,
    create_initial_state,
)

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)

from adaptive_agentic_rag.orchestration.adaptive_retry_policy import (
    AdaptiveRetryPolicy,
    RetryAction,
)

from adaptive_agentic_rag.orchestration.corpus_source_availability import (
    CorpusSourceAvailability,
)


# ============================================================
# Shared lazy corpus-source catalog
#
# No corpus file is loaded during module import.
#
# The component loads the source catalog only if the retry
# policy actually encounters a partial explicit-source miss.
#
# One graph process therefore builds the catalog once.
# ============================================================

_CORPUS_SOURCE_AVAILABILITY = (
    CorpusSourceAvailability()
)


# ============================================================
# Conditional routing after evidence grading
# ============================================================

def route_after_evidence(
    state: AgentState,
    *,
    source_availability=None,
) -> str:
    """
    Decide what the graph should do after evidence grading.

        grade_evidence
              ↓
        AdaptiveRetryPolicy
              │
        ┌─────┼─────┐
        │     │     │
        ▼     ▼     ▼
    generate retry abstain

    Retry now requires:

    1. evidence insufficiency,
    2. remaining retry budget,
    3. partial explicit-source coverage,
    4. missing source actually existing in the corpus.

    This prevents structurally impossible retrieval retries.
    """

    evidence_sufficient = (
        state.get(
            "evidence_sufficient"
        )
    )


    if (
        evidence_sufficient
        is None
    ):

        raise ValueError(
            "Cannot route after evidence grading "
            "without evidence_sufficient."
        )


    retry_count = (
        state.get(
            "retry_count",
            0,
        )
        or 0
    )


    max_retries = (
        state.get(
            "max_retries",
            0,
        )
        or 0
    )


    evidence_reasons = list(
        state.get(
            "evidence_reasons",
            [],
        )
        or []
    )


    # ========================================================
    # Dependency injection
    #
    # LangGraph invokes this function with only `state`,
    # therefore production uses the shared lazy catalog.
    #
    # Tests can inject a deterministic fake availability
    # implementation through the optional keyword argument.
    # ========================================================

    if (
        source_availability
        is None
    ):

        source_availability = (
            _CORPUS_SOURCE_AVAILABILITY
        )


    retry_policy = (
        AdaptiveRetryPolicy(
            max_retries=
                int(
                    max_retries
                ),

            source_availability=
                source_availability,
        )
    )


    decision = (
        retry_policy.decide(
            evidence_sufficient=
                bool(
                    evidence_sufficient
                ),

            retry_count=
                int(
                    retry_count
                ),

            evidence_reasons=
                evidence_reasons,
        )
    )


    if (
        decision.action
        ==
        RetryAction.GENERATE
    ):

        return "generate"


    if (
        decision.action
        ==
        RetryAction.RETRY
    ):

        return "rewrite"


    if (
        decision.action
        ==
        RetryAction.ABSTAIN
    ):

        return "abstain"


    raise RuntimeError(
        (
            "Unsupported retry-policy action: "
            f"{decision.action!r}"
        )
    )


# ============================================================
# Main Graph
# ============================================================

class AdaptiveRAGGraph:

    def __init__(
        self,
    ):

        self.nodes = (
            RAGNodes()
        )


        builder = (
            StateGraph(
                AgentState
            )
        )


        # ====================================================
        # Nodes
        # ====================================================

        builder.add_node(
            "route_query",
            self.nodes.route_query,
        )


        builder.add_node(
            "retrieve",
            self.nodes.retrieve,
        )


        builder.add_node(
            "build_context",
            self.nodes.build_context,
        )


        builder.add_node(
            "grade_evidence",
            self.nodes.grade_evidence,
        )


        builder.add_node(
            "rewrite_query",
            self.nodes.rewrite_query,
        )


        builder.add_node(
            "generate",
            self.nodes.generate,
        )


        builder.add_node(
            "abstain",
            self.nodes.abstain,
        )


        builder.add_node(
            "grade_answer",
            self.nodes.grade_answer,
        )


        # ====================================================
        # Main forward path
        # ====================================================

        builder.add_edge(
            START,
            "route_query",
        )


        builder.add_edge(
            "route_query",
            "retrieve",
        )


        builder.add_edge(
            "retrieve",
            "build_context",
        )


        builder.add_edge(
            "build_context",
            "grade_evidence",
        )


        # ====================================================
        # Adaptive evidence routing
        # ====================================================

        builder.add_conditional_edges(
            "grade_evidence",

            route_after_evidence,

            {
                "generate":
                    "generate",

                "rewrite":
                    "rewrite_query",

                "abstain":
                    "abstain",
            },
        )


        # ====================================================
        # Self-correction loop
        #
        # Only recoverable, corpus-available structural misses
        # may enter this loop.
        # ====================================================

        builder.add_edge(
            "rewrite_query",
            "retrieve",
        )


        # ====================================================
        # Answer grading
        # ====================================================

        builder.add_edge(
            "generate",
            "grade_answer",
        )


        builder.add_edge(
            "abstain",
            "grade_answer",
        )


        builder.add_edge(
            "grade_answer",
            END,
        )


        self.graph = (
            builder.compile()
        )


    # ========================================================
    # Execute
    # ========================================================

    def run(
        self,
        query: str,
        max_retries: int = 1,
    ) -> AgentState:

        initial_state = (
            create_initial_state(
                query=
                    query,

                max_retries=
                    max_retries,
            )
        )


        final_state = (
            self.graph.invoke(
                initial_state
            )
        )


        return final_state


    # ========================================================
    # Cleanup
    # ========================================================

    def close(
        self,
    ):

        self.nodes.close()