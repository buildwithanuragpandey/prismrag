from adaptive_agentic_rag.agents.self_correction import (
    SelfCorrectionController
)


def main():

    controller = SelfCorrectionController(
        max_retries=1
    )


    try:

        query = (
            "Compare Amazon and Walmart deals"
        )


        claims = [

            {
                "id": "claim_1",
                "text": (
                    "Best Buy matches Amazon's pricing "
                    "on qualifying items."
                )
            },

            {
                "id": "claim_2",
                "text": (
                    "Walmart does not price-match "
                    "competitors' products."
                )
            },

            {
                "id": "claim_3",
                "text": (
                    "Apple does not offer a universal "
                    "price-matching policy."
                )
            },

            {
                "id": "claim_4",
                "text": (
                    "Amazon typically hosts Black Friday "
                    "sales on electronics."
                )
            },

            {
                "id": "claim_5",
                "text": (
                    "Many retailers participate in "
                    "Black Friday sales."
                )
            },

            {
                "id": "claim_6",
                "text": (
                    "Amazon Prime members receive "
                    "free games throughout the year."
                )
            }

        ]


        #
        # Reuse the reranker that is
        # already loaded inside AdaptiveRetriever.
        #

        reranker = (
            controller
            .retriever
            .reranked
            .reranker
        )


        results = reranker.rerank(

            query=query,

            documents=claims,

            top_k=len(claims)

        )


        print(
            "\n===== QUERY ====="
        )

        print(query)


        print(
            "\n===== CROSS-ENCODER RELEVANCE ====="
        )


        for item in results:

            print(
                "\nScore:",
                round(
                    item["rerank_score"],
                    4
                )
            )

            print(
                "Claim:",
                item["text"]
            )


    finally:

        controller.close()


if __name__ == "__main__":
    main()