from adaptive_agentic_rag.retrieval.reranked_retriever import (
    RerankedRetriever
)

from adaptive_agentic_rag.retrieval.dense_retriever import DenseRetriever


dense = DenseRetriever()


retriever = RerankedRetriever(
    dense_retriever=dense,
    hybrid_top_k=20
)


results = retriever.search(
    "amazon black friday deals"
)


print("\n===== FINAL MMR OUTPUT =====")


for item in results:

    print(
        item["id"],
        "vector:",
        "vector" in item,
        "score:",
        item["score"]
    )


retriever.close()