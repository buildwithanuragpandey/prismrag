import numpy as np

from adaptive_agentic_rag.retrieval.adaptive_retriever import (
    AdaptiveRetriever
)

from adaptive_agentic_rag.generation.context_builder import (
    ContextBuilder
)

from adaptive_agentic_rag.generation.claim_grounder import (
    ClaimGrounder
)


def main():

    query = (
        "Compare Amazon and Walmart deals"
    )

    retriever = AdaptiveRetriever()

    builder = ContextBuilder(
        max_words=1800,
        max_chunks=5,
        max_chunks_per_document=2
    )

    #
    # اول باید مدل ساخته شود
    #
    grounder = ClaimGrounder()


    try:

        #
        # ==========================================
        # 1. NLI sanity check
        # ==========================================
        #

        test_pairs = [

            (
                "Walmart does not price-match the prices of its competitors.",
                "Walmart does not price-match competitors."
            ),

            (
                "Walmart does not price-match the prices of its competitors.",
                "Walmart price-matches competitors."
            ),

            (
                "Amazon offers free games and DLC for Prime members throughout the year.",
                "Amazon Prime members receive free games and DLC throughout the year."
            ),

            (
                "Amazon sells many different products.",
                "Amazon gives 90 percent discounts every Friday."
            )

        ]


        scores = grounder.model.predict(
            test_pairs,
            apply_softmax=True
        )


        labels = [
            "contradiction",
            "entailment",
            "neutral"
        ]


        print(
            "\n===== NLI SANITY CHECK ====="
        )


        for pair, score in zip(
            test_pairs,
            scores
        ):

            predicted_index = int(
                np.argmax(score)
            )


            print(
                "\nPremise:",
                pair[0]
            )

            print(
                "Claim:",
                pair[1]
            )

            print(
                "Probabilities:",
                {
                    labels[i]: round(
                        float(score[i]),
                        4
                    )
                    for i in range(3)
                }
            )

            print(
                "Prediction:",
                labels[predicted_index]
            )


        #
        # ==========================================
        # 2. Real retrieved context
        # ==========================================
        #

        retrieval_output = retriever.search(
            query,
            top_k=10
        )


        context = builder.build(
            retrieval_output["results"]
        )


        answer = """
- Walmart does not price-match competitors' products, limiting its ability to compete effectively in price wars.
- Amazon Prime members receive free games and DLC throughout the year, enhancing their shopping experience.
- Amazon gives 90 percent discounts every Friday.
""".strip()


        result = grounder.ground(
            answer=answer,
            context=context
        )


        print(
            "\n\n===== TEST CLAIMS ====="
        )

        print(answer)


        print(
            "\n===== CLAIM GROUNDING ====="
        )


        for claim in result.claims:

            print(
                "\nClaim:",
                claim.claim
            )

            print(
                "Supported:",
                claim.supported
            )

            print(
                "Citation:",
                claim.citation_id
            )

            print(
                "Label:",
                claim.label
            )

            print(
                "Entailment score:",
                claim.entailment_score
            )
            print(
                "Supporting text:",
                claim.supporting_text
            )            


        print(
            "\n===== SUMMARY ====="
        )

        print(
            "Supported:",
            result.supported_count
        )

        print(
            "Unsupported:",
            result.unsupported_count
        )


    finally:

        retriever.close()


if __name__ == "__main__":
    main()