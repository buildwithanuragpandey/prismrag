import time

from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever
)


def main():

    query = (
        "What are the best Amazon Black Friday deals?"
    )


    print("Loading components...")


    dense = DenseRetriever()



    #
    # Embedding latency
    #

    start = time.perf_counter()


    query_vector = dense.embedder.encode_queries(
        [query]
    )[0]


    end = time.perf_counter()


    print(
        "Embedding:",
        round(
            (end-start)*1000,
            2
        ),
        "ms"
    )



    #
    # Qdrant latency
    #

    start = time.perf_counter()


    results = dense.search_by_vector(
        query_vector,
        top_k=20
    )


    end = time.perf_counter()


    print(
        "Qdrant search:",
        round(
            (end-start)*1000,
            2
        ),
        "ms"
    )



    dense.close()



if __name__ == "__main__":
    main()