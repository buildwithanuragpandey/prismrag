from adaptive_agentic_rag.retrieval.hybrid_retriever import HybridRetriever


retriever = HybridRetriever()


results = retriever.search(
    "Amazon deals",
    top_k=5
)


for r in results:

    print(r.keys())

    print(
        r["metadata"]
    )


retriever.close()