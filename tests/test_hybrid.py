from adaptive_agentic_rag.retrieval.hybrid_retriever import (
    HybridRetriever
)



retriever = HybridRetriever()



results = retriever.search(
    "Amazon Cyber Monday deals",
    top_k=5
)



for item in results:


    print(
        item["id"],
        item["score"],
        item["metadata"]["title"]
    )



retriever.close()