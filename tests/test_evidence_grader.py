from adaptive_agentic_rag.retrieval.adaptive_retriever import (
    AdaptiveRetriever
)

from adaptive_agentic_rag.generation.context_builder import (
    ContextBuilder,
    BuiltContext
)

from adaptive_agentic_rag.agents.evidence_grader import (
    EvidenceGrader
)


def print_grade(
    title,
    grade
):

    print(
        f"\n===== {title} ====="
    )

    print(
        "Sufficient:",
        grade.sufficient
    )

    print(
        "Evidence score:",
        grade.evidence_score
    )

    print(
        "Query coverage:",
        grade.query_term_coverage
    )

    print(
        "Unique documents:",
        grade.unique_documents
    )

    print(
        "Chunks:",
        grade.chunk_count
    )

    print(
        "Weak citations:",
        grade.weak_citations
    )

    print(
        "Reasons:"
    )

    for reason in grade.reasons:

        print(
            "-",
            reason
        )


def main():

    query = (
        "Compare Amazon and Walmart deals"
    )


    retriever = (
        AdaptiveRetriever()
    )


    builder = ContextBuilder(

        max_words=1800,

        max_chunks=5,

        max_chunks_per_document=2

    )


    grader = (
        EvidenceGrader()
    )


    try:

        #
        # ==========================
        # Real retrieved context
        # ==========================
        #

        retrieval_output = (
            retriever.search(
                query,
                top_k=10
            )
        )


        decision = (
            retrieval_output[
                "decision"
            ]
        )


        context = builder.build(

            retrieval_output[
                "results"
            ]

        )


        grade = grader.grade(

            query=query,

            context=context,

            query_type=(
                decision[
                    "query_type"
                ]
            )

        )


        print(
            "\n===== QUERY ====="
        )

        print(
            query
        )


        print(
            "\n===== ROUTING ====="
        )

        print(
            decision
        )


        print_grade(

            "FULL CONTEXT GRADE",

            grade

        )


        #
        # ==========================
        # Artificial insufficient
        # context
        # ==========================
        #

        if context.items:

            first_item = (
                context.items[0]
            )


            insufficient_context = (
                BuiltContext(

                    text=(
                        first_item.text
                    ),

                    items=[
                        first_item
                    ],

                    total_words=len(
                        first_item.text.split()
                    )

                )
            )


            insufficient_grade = (
                grader.grade(

                    query=query,

                    context=(
                        insufficient_context
                    ),

                    query_type=(
                        decision[
                            "query_type"
                        ]
                    )

                )
            )


            print_grade(

                "INSUFFICIENT CONTEXT TEST",

                insufficient_grade

            )


    finally:

        retriever.close()


if __name__ == "__main__":
    main()