from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever
)



def test_dense_search():


    retriever = DenseRetriever()


    results = retriever.search(

        "Who founded FTX?",

        top_k=3

    )


    assert len(results) == 3


    assert "text" in results[0]