from adaptive_agentic_rag.retrieval.adaptive_retriever import (
    AdaptiveRetriever
)

from adaptive_agentic_rag.generation.context_builder import (
    ContextBuilder
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


    try:

        retrieval_output = (
            retriever.search(
                query,
                top_k=10
            )
        )


        context = builder.build(
            retrieval_output[
                "results"
            ]
        )


        print(
            "\n===== ROUTING DECISION ====="
        )

        print(
            retrieval_output[
                "decision"
            ]
        )


        print(
            "\n===== CONTEXT STATS ====="
        )

        print(
            "Chunks:",
            len(context.items)
        )

        print(
            "Words:",
            context.total_words
        )


        print(
            "\n===== SOURCES ====="
        )


        for item in context.items:

            print(
                f"[{item.citation_id}] "
                f"{item.document_id} | "
                f"{item.title}"
            )


        print(
            "\n===== FINAL CONTEXT ====="
        )

        print(
            context.text
        )


    finally:

        retriever.close()


if __name__ == "__main__":
    main()