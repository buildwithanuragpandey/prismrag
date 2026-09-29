from adaptive_agentic_rag.retrieval.hybrid_retriever import (
    HybridRetriever
)

from adaptive_agentic_rag.retrieval.reranker import (
    BGEReranker
)



query = (
    "Amazon Cyber Monday deals"
)


hybrid = HybridRetriever()


reranker = BGEReranker()



results = hybrid.search(
    query,
    top_k=20
)


print("\n===== BEFORE RERANK =====")


for r in results[:5]:

    print(
        r["id"],
        r["score"],
        r["metadata"]["title"]
    )



reranked = reranker.rerank(
    query,
    results,
    top_k=5
)



print("\n===== AFTER RERANK =====")


for r in reranked:

    print(
        r["id"],
        r["rerank_score"],
        r["metadata"]["title"]
    )



hybrid.close()