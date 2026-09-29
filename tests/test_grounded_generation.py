from adaptive_agentic_rag.agents.self_correction import (
    SelfCorrectionController
)

from adaptive_agentic_rag.generation.generator import (
    GroundedGenerator
)


def print_generation(
    retrieval_result,
    generation_result
):

    print(
        "\n===== RETRIEVAL ====="
    )

    print(
        "Rounds:",
        retrieval_result.retrieval_rounds
    )

    print(
        "Rewritten:",
        retrieval_result.rewritten
    )

    print(
        "Evidence sufficient:",
        retrieval_result.evidence_sufficient
    )

    print(
        "Evidence score:",
        retrieval_result.grade.evidence_score
    )


    print(
        "\n===== GENERATION ====="
    )

    print(
        "Abstained:",
        generation_result.abstained
    )

    print(
        "Model:",
        generation_result.model_name
    )

    print(
        "\nAnswer:"
    )

    print(
        generation_result.answer
    )


    print(
        "\n===== CITATIONS ====="
    )

    print(
        "Cited IDs:",
        generation_result.cited_ids
    )

    print(
        "Invalid IDs:",
        generation_result.invalid_citation_ids
    )

    print(
        "Citation valid:",
        generation_result.citation_valid
    )


def main():

    controller = (
        SelfCorrectionController(
            max_retries=1
        )
    )


    generator = GroundedGenerator(

        reranker=(
            controller
            .retriever
            .reranked
            .reranker
        ),

        max_relevant_claims=2

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


            retrieval_result = (
                controller.run(

                    query=query,

                    top_k=10

                )
            )


            generation_result = (
                generator.generate(

                    #
                    # Always original user question
                    #

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

                    max_new_tokens=250

                )
            )


            print_generation(

                retrieval_result,

                generation_result

            )
            print(
                "\n===== CLAIM GROUNDING ====="
            )

            print(
                "Supported claims:",
                generation_result.supported_claims
            )

            print(
                "Unsupported claims:",
                generation_result.unsupported_claims
            )
            print(
                "Relevant claims:",
                generation_result.relevant_claims
            )

            print(
                "Filtered irrelevant claims:",
                generation_result.filtered_irrelevant_claims
            )

    finally:

        controller.close()


if __name__ == "__main__":
    main()