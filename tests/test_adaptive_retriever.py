from adaptive_agentic_rag.retrieval.adaptive_retriever import (
    AdaptiveRetriever
)



retriever = AdaptiveRetriever()



queries = [

    "What is Amazon?",

    "Compare Amazon and Walmart deals",

    "Summarize all Black Friday deals"

]



for query in queries:


    print("\n================")

    print("QUERY:")

    print(query)



    result = retriever.search(
        query,
        top_k=5
    )


    print("\nDECISION:")

    print(
        result["decision"]
    )


    print("\nRESULTS:")


    for item in result["results"]:

        print(
            item["id"],
            item["metadata"]["title"]
        )



retriever.close()