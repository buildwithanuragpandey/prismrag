from typing import Any

from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state for one Adaptive Agentic RAG execution.

    Important:
    This contains per-request data only.

    Models, retrievers, rerankers, graders,
    and other shared services do NOT belong here.
    """

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    original_query: str

    current_query: str


    # --------------------------------------------------
    # Query routing
    # --------------------------------------------------

    query_type: str | None

    retrieval_strategy: str | None

    use_reranker: bool

    use_mmr: bool


    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    retrieved_results: list[dict[str, Any]]

    # Missing publishers explicitly targeted during a
    # structurally approved retry.
    #
    # Empty during normal first-pass retrieval.
    retry_target_sources: list[str]


    # --------------------------------------------------
    # Context
    # --------------------------------------------------

    context: Any | None


    # --------------------------------------------------
    # Evidence grading
    # --------------------------------------------------

    evidence_sufficient: bool | None

    evidence_score: float | None

    evidence_reasons: list[str]


    # --------------------------------------------------
    # Self-correction
    # --------------------------------------------------

    retry_count: int

    max_retries: int

    rewritten: bool


    # --------------------------------------------------
    # Generation
    # --------------------------------------------------

    generation_result: Any | None

    generation_model_name: str | None

    raw_answer: str | None

    final_answer: str | None

    abstained: bool


    # --------------------------------------------------
    # Claim verification
    # --------------------------------------------------

    supported_claims: int

    unsupported_claims: int

    relevant_claims: int

    filtered_irrelevant_claims: int


    # --------------------------------------------------
    # Citation validation
    # --------------------------------------------------

    citation_valid: bool | None


    # --------------------------------------------------
    # Final answer grading
    # --------------------------------------------------

    answer_grade: Any | None

    answer_passed: bool | None

    answer_relevance_score: float | None

    answer_grade_reasons: list[str]


    # --------------------------------------------------
    # Error handling / observability
    # --------------------------------------------------

    error: str | None


def create_initial_state(
    query: str,
    max_retries: int = 1,
) -> AgentState:

    return AgentState(
        original_query=
            query,

        current_query=
            query,

        query_type=
            None,

        retrieval_strategy=
            None,

        use_reranker=
            False,

        use_mmr=
            False,

        retrieved_results=
            [],

        retry_target_sources=
            [],

        context=
            None,

        evidence_sufficient=
            None,

        evidence_score=
            None,

        evidence_reasons=
            [],

        retry_count=
            0,

        max_retries=
            max_retries,

        rewritten=
            False,

        generation_result=
            None,

        generation_model_name=
            None,

        raw_answer=
            None,

        final_answer=
            None,

        abstained=
            False,

        supported_claims=
            0,

        unsupported_claims=
            0,

        relevant_claims=
            0,

        filtered_irrelevant_claims=
            0,

        citation_valid=
            None,

        answer_grade=
            None,

        answer_passed=
            None,

        answer_relevance_score=
            None,

        answer_grade_reasons=
            [],

        error=
            None,
    )