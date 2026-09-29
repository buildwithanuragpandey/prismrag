from adaptive_agentic_rag.agents.self_correction import (
    SelfCorrectionController
)

from adaptive_agentic_rag.generation.generator import (
    GroundedGenerator
)

from adaptive_agentic_rag.agents.answer_grader import (
    AnswerGrader
)


def print_grade(
    result
):

    print(
        "\n===== ANSWER GRADE ====="
    )

    print(
        "Passed:",
        result.passed
    )

    print(
        "Correct abstention:",
        result.correct_abstention
    )

    print(
        "Citation valid:",
        result.citation_valid
    )

    print(
        "Supported claim ratio:",
        result.supported_claim_ratio
    )

    print(
        "Relevance score:",
        result.relevance_score
    )

    print(
        "Reasons:"
    )


    for reason in result.reasons:

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


    generator = (
        GroundedGenerator()
    )


    #
    # IMPORTANT:
    # Reuse the embedding model
    # already loaded by DenseRetriever.
    #

    grader = AnswerGrader(

        embedder=(
            controller
            .retriever
            .dense
            .embedder
        )

    )


    queries = [

        (
            "Compare Amazon "
            "and Walmart deals"
        ),

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


            #
            # Retrieval + self-correction
            #

            retrieval_result = (
                controller.run(

                    query=query,

                    top_k=10

                )
            )


            #
            # Generation + claim grounding
            #

            generation_result = (
                generator.generate(

                    query=(
                        retrieval_result
                        .original_query
                    ),

                    context=(
                        retrieval_result
                        .context
                    ),

                    evidence_sufficient=(
                        retrieval_result
                        .evidence_sufficient
                    ),

                    max_new_tokens=160

                )
            )


            print(
                "\n===== FINAL ANSWER ====="
            )

            print(
                generation_result.answer
            )


            print(
                "\nSupported:",
                generation_result.supported_claims
            )

            print(
                "Unsupported:",
                generation_result.unsupported_claims
            )


            #
            # Final answer grading
            #

            grade = grader.grade(

                query=(
                    retrieval_result
                    .original_query
                ),

                generation_result=(
                    generation_result
                ),

                evidence_sufficient=(
                    retrieval_result
                    .evidence_sufficient
                )

            )


            print_grade(
                grade
            )


    finally:

        controller.close()


if __name__ == "__main__":
    main()