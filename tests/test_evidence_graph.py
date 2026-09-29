from adaptive_agentic_rag.orchestration.graph import (
    AdaptiveRAGGraph
)


def print_state_summary(
    title: str,
    state: dict
):

    print(
        "\n"
        "================================"
    )

    print(title)


    # ==================================================
    # Query
    # ==================================================

    print(
        "\n===== QUERY ====="
    )

    print(
        "Original:",
        state[
            "original_query"
        ]
    )

    print(
        "Current:",
        state[
            "current_query"
        ]
    )


    # ==================================================
    # Routing
    # ==================================================

    print(
        "\n===== ROUTING ====="
    )

    print(
        "Query type:",
        state[
            "query_type"
        ]
    )

    print(
        "Retrieval strategy:",
        state[
            "retrieval_strategy"
        ]
    )

    print(
        "Use reranker:",
        state[
            "use_reranker"
        ]
    )

    print(
        "Use MMR:",
        state[
            "use_mmr"
        ]
    )


    # ==================================================
    # Retrieval
    # ==================================================

    print(
        "\n===== RETRIEVAL ====="
    )

    print(
        "Retrieved chunks:",
        len(
            state[
                "retrieved_results"
            ]
        )
    )


    # ==================================================
    # Evidence
    # ==================================================

    print(
        "\n===== EVIDENCE ====="
    )

    print(
        "Sufficient:",
        state[
            "evidence_sufficient"
        ]
    )

    print(
        "Score:",
        state[
            "evidence_score"
        ]
    )

    print(
        "Reasons:",
        state[
            "evidence_reasons"
        ]
    )


    # ==================================================
    # Self-correction
    # ==================================================

    print(
        "\n===== SELF CORRECTION ====="
    )

    print(
        "Retry count:",
        state[
            "retry_count"
        ]
    )

    print(
        "Max retries:",
        state[
            "max_retries"
        ]
    )

    print(
        "Rewritten:",
        state[
            "rewritten"
        ]
    )


    # ==================================================
    # Generation
    # ==================================================

    print(
        "\n===== GENERATION ====="
    )

    print(
        "Abstained:",
        state[
            "abstained"
        ]
    )

    print(
        "Model:",
        state[
            "generation_model_name"
        ]
    )

    print(
        "Citation valid:",
        state[
            "citation_valid"
        ]
    )


    print(
        "\nFinal answer:"
    )

    print(
        state[
            "final_answer"
        ]
    )


    # ==================================================
    # Claims
    # ==================================================

    print(
        "\n===== CLAIMS ====="
    )
    print(
        "Supported:",
        state[
            "supported_claims"
        ]
    )

    print(
        "Unsupported:",
        state[
            "unsupported_claims"
        ]
    )

    print(
        "Relevant:",
        state[
            "relevant_claims"
        ]
    )

    print(
        "Filtered irrelevant:",
        state[
            "filtered_irrelevant_claims"
        ]
    )


    # ==================================================
    # Answer grading
    # ==================================================

    print(
        "\n===== ANSWER GRADE ====="
    )

    print(
        "Passed:",
        state[
            "answer_passed"
        ]
    )

    print(
        "Relevance score:",
        state[
            "answer_relevance_score"
        ]
    )

    print(
        "Reasons:",
        state[
            "answer_grade_reasons"
        ]
    )


def main():

    workflow = (
        AdaptiveRAGGraph()
    )


    try:

        # ==================================================
        # CASE 1
        #
        # Expected:
        #
        # sufficient evidence
        # -> generation
        # -> claim verification
        # -> relevance
        # -> citations
        # -> answer grading
        # ==================================================

        query_1 = (
            "Compare Amazon and Walmart deals"
        )


        state_1 = (
            workflow.run(

                query=query_1,

                max_retries=1
            )
        )


        print_state_summary(

            title=(
                "CASE 1 — "
                "FULL GROUNDED ANSWER"
            ),

            state=state_1
        )


        assert (
            state_1[
                "evidence_sufficient"
            ]
            is True
        )


        assert (
            state_1[
                "retry_count"
            ]
            ==
            0
        )


        assert (
            state_1[
                "rewritten"
            ]
            is False
        )


        assert (
            state_1[
                "generation_result"
            ]
            is not None
        )


        assert (
            state_1[
                "abstained"
            ]
            is False
        )


        assert (
            state_1[
                "final_answer"
            ]
            is not None
        )


        assert (
            state_1[
                "final_answer"
            ].strip()
            !=
            ""
        )


        assert (
            state_1[
                "answer_grade"
            ]
            is not None
        )


        # ==================================================
        # CASE 2
        #
        # Expected:
        #
        # insufficient
        # -> rewrite
        # -> retrieve again
        # -> still insufficient
        # -> abstain
        # -> answer grader
        # ==================================================

        query_2 = (
            "Compare Amazon and AcmeMart "
            "loyalty cashback shipping deals"
        )


        state_2 = (
            workflow.run(

                query=query_2,

                max_retries=1
            )
        )


        print_state_summary(

            title=(
                "CASE 2 — "
                "SELF-CORRECTION + ABSTENTION"
            ),

            state=state_2
        )


        assert (
            state_2[
                "evidence_sufficient"
            ]
            is False
        )


        assert (
            state_2[
                "retry_count"
            ]
            ==
            1
        )


        assert (
            state_2[
                "rewritten"
            ]
            is True
        )


        assert (
            state_2[
                "current_query"
            ]
            !=
            state_2[
                "original_query"
            ]
        )


        assert (
            state_2[
                "generation_result"
            ]
            is not None
        )


        assert (
            state_2[
                "abstained"
            ]
            is True
        )


        assert (
            state_2[
                "final_answer"
            ]
            is not None
        )


        assert (
            state_2[
                "answer_grade"
            ]
            is not None
        )


        print(
            "\n"
            "================================"
        )

        print(
            "END-TO-END LANGGRAPH: OK"
        )


    finally:

        workflow.close()


if __name__ == "__main__":

    main()