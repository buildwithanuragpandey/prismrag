from adaptive_agentic_rag.retrieval.bm25_retriever import BM25Retriever


retriever = BM25Retriever()


results = retriever.search(
    "Amazon Cyber Monday deals",
    top_k=5
)


for r in results:
    print(
        r["id"],
        r["score"]
    )