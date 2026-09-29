from adaptive_agentic_rag.agents.query_router import (
    QueryRouter
)



router = QueryRouter()



queries = [

    "What is Amazon?",

    "Compare Amazon and Walmart deals",

    "Explain the relationship between Apple and Amazon products",

    "Summarize all Black Friday deals"

]



for q in queries:


    result = router.route(q)


    print("\nQUERY:")
    print(q)

    print(result)