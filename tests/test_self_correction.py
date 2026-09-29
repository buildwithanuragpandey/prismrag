from adaptive_agentic_rag.agents.self_correction import (
    SelfCorrectionController
)


def print_result(
    result
):

    print(
        "\n===== FINAL RESULT ====="
    )

    print(
        "Original query:",
        result.original_query
    )

    print(
        "Final query:",
        result.final_query
    )

    print(
        "Retrieval rounds:",
        result.retrieval_rounds
    )

    print(
        "Rewritten:",
        result.rewritten
    )

    print(
        "Evidence sufficient:",
        result.evidence_sufficient
    )

    print(
        "Final evidence score:",
        result.grade.evidence_score
    )


    print(
        "\n===== HISTORY ====="
    )


    for round_info in result.history:

        print(
            f"\nROUND "
            f"{round_info.round_number}"
        )

        print(
            "Query:",
            round_info.query
        )

        print(
            "Routing:",
            round_info.decision
        )

        print(
            "Sufficient:",
            round_info.grade.sufficient
        )

        print(
            "Evidence score:",
            round_info.grade.evidence_score
        )

        print(
            "Coverage:",
            round_info.grade.query_term_coverage
        )

        print(
            "Documents:",
            round_info.grade.unique_documents
        )

        print(
            "Reasons:"
        )


        for reason in (
            round_info.grade.reasons
        ):

            print(
                "-",
                reason
            )


def main():

    controller = (
        SelfCorrectionController(
            max_retries=1
        )
    )


    queries = [

        #
        # Should normally succeed
        # without rewrite
        #

        (
            "Compare Amazon "
            "and Walmart deals"
        ),


        #
        # Intentionally difficult.
        # Useful for exercising
        # insufficient-evidence path.
        #

        (
            "Compare Amazon and "
            "AcmeMart loyalty cashback "
            "shipping deals"
        )

    ]


    try:

        for query in queries:

            print(
                "\n\n"
                "================================"
            )

            print(
                "QUERY:"
            )

            print(
                query
            )


            result = controller.run(

                query=query,

                top_k=10

            )


            print_result(
                result
            )


    finally:

        controller.close()


if __name__ == "__main__":
    main()