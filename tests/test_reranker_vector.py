from adaptive_agentic_rag.retrieval.hybrid_retriever import HybridRetriever
from adaptive_agentic_rag.retrieval.reranker import BGEReranker


hybrid = HybridRetriever()

reranker = BGEReranker()


query = "amazon black friday deals"


candidates = hybrid.search(
    query,
    top_k=20
)


print("\n===== BEFORE RERANK =====")

for x in candidates[:3]:

    print(
        x.keys()
    )



reranked = reranker.rerank(
    query,
    candidates,
    top_k=5
)


print("\n===== AFTER RERANK =====")

for x in reranked:

    print(
        x.keys()
    )


hybrid.close()