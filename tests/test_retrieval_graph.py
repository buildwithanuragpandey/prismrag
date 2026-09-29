from adaptive_agentic_rag.orchestration.graph import (
    AdaptiveRAGGraph
)


def main():

    workflow = (
        AdaptiveRAGGraph()
    )


    try:

        query = (
            "Compare Amazon and Walmart deals"
        )


        final_state = (
            workflow.run(
                query=query
            )
        )


        print(
            "\n"
            "================================"
        )

        print(
            "QUERY:"
        )

        print(
            final_state[
                "original_query"
            ]
        )


        print(
            "\n"
            "===== ROUTING ====="
        )

        print(
            "Query type:",
            final_state[
                "query_type"
            ]
        )

        print(
            "Retrieval strategy:",
            final_state[
                "retrieval_strategy"
            ]
        )

        print(
            "Use reranker:",
            final_state[
                "use_reranker"
            ]
        )

        print(
            "Use MMR:",
            final_state[
                "use_mmr"
            ]
        )


        print(
            "\n"
            "===== RETRIEVAL ====="
        )

        results = (
            final_state[
                "retrieved_results"
            ]
        )


        print(
            "Retrieved chunks:",
            len(results)
        )


        for index, item in enumerate(
            results[:5],
            start=1
        ):

            print(
                "\n"
                f"Result {index}"
            )

            print(
                "Document:",
                item.get(
                    "document_id"
                )
            )

            print(
                "Title:",
                item.get(
                    "title"
                )
            )

            print(
                "Score:",
                item.get(
                    "score"
                )
            )


        print(
            "\n"
            "===== CONTEXT ====="
        )


        context = (
            final_state[
                "context"
            ]
        )


        print(
            "Context type:",
            type(
                context
            ).__name__
        )


        print(
            "Chunks:",
            len(
                context.items
            )
        )


        print(
            "Total words:",
            context.total_words
        )


        print(
            "\n"
            "===== STATE CHECK ====="
        )


        assert (
            final_state[
                "original_query"
            ]
            ==
            query
        )


        assert (
            final_state[
                "current_query"
            ]
            ==
            query
        )


        assert (
            final_state[
                "query_type"
            ]
            is not None
        )


        assert (
            len(
                final_state[
                    "retrieved_results"
                ]
            )
            >
            0
        )


        assert (
            final_state[
                "context"
            ]
            is not None
        )


        print(
            "Graph execution: OK"
        )


    finally:

        workflow.close()


if __name__ == "__main__":

    main()