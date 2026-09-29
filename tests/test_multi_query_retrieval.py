from adaptive_agentic_rag.retrieval.query_decomposer import (
    QueryDecomposer
)

from adaptive_agentic_rag.retrieval.multi_query_retriever import (
    MultiQueryRetriever
)


# ============================================================
# Fake Hybrid Retriever
# ============================================================

class FakeHybridRetriever:

    def search(
        self,
        query,
        top_k=20
    ):

        query_lower = (
            query.lower()
        )


        #
        # Simulate semantic dilution.
        #
        # The original monster query retrieves only
        # one useful evidence document.
        #

        if (
            len(
                query.split()
            )
            >
            30
        ):

            return [

                {
                    "id":
                        "chunk_trial",

                    "document_id":
                        "doc_trial",

                    "text":
                        "trial legal narratives",

                    "score":
                        1.0
                }
            ]


        results = []


        # ----------------------------------------------------
        # Trial facet
        # ----------------------------------------------------

        if (
            "trial"
            in
            query_lower
        ):

            results.append(

                {
                    "id":
                        "chunk_trial",

                    "document_id":
                        "doc_trial",

                    "text":
                        "trial legal narratives",

                    "score":
                        1.0
                }
            )


        # ----------------------------------------------------
        # Financial discrepancy facet
        # ----------------------------------------------------

        if (
            "financial discrepancy"
            in
            query_lower
        ):

            results.append(

                {
                    "id":
                        "chunk_financial",

                    "document_id":
                        "doc_financial",

                    "text":
                        "financial discrepancy evidence",

                    "score":
                        1.0
                }
            )


        # ----------------------------------------------------
        # Fraud facet
        # ----------------------------------------------------

        if (
            "fraud"
            in
            query_lower
        ):

            results.append(

                {
                    "id":
                        "chunk_fraud",

                    "document_id":
                        "doc_fraud",

                    "text":
                        "intentional fraud evidence",

                    "score":
                        1.0
                }
            )


        # ----------------------------------------------------
        # Investor comparison facet
        # ----------------------------------------------------

        if (
            "prominent investor"
            in
            query_lower
        ):

            results.append(

                {
                    "id":
                        "chunk_investor",

                    "document_id":
                        "doc_investor",

                    "text":
                        "prominent investor comparison",

                    "score":
                        1.0
                }
            )


        return results[
            :top_k
        ]


# ============================================================
# Test query
# ============================================================

query = (
    "Who is the individual whose trial involves contrasting "
    "legal narratives presented to a jury, "
    "as reported by Fortune, "
    "was previously likened to a prominent investor but not "
    "by TechCrunch, "
    "admitted to being aware of a significant financial "
    "discrepancy after a judge's intervention according to "
    "The Verge, "
    "and is accused of intentional fraud for personal gain "
    "as per allegations mentioned in a second TechCrunch article?"
)


# ============================================================
# Query decomposition
# ============================================================

decomposer = (
    QueryDecomposer()
)


queries = (
    decomposer.decompose(
        query
    )
)


print(
    "\n===== QUERY DECOMPOSITION ====="
)


for index, item in enumerate(
    queries,
    start=1
):

    print(
        index,
        item
    )


# ============================================================
# Decomposition assertions
# ============================================================

assert (
    queries[0]
    ==
    query
)


#
# This query is intentionally very long.
#
# It should activate the extended facet budget:
#
# original + up to 4 facets.
#

assert (
    len(
        queries
    )
    >=
    4
)


assert (
    len(
        queries
    )
    <=
    5
)


query_text = (
    "\n".join(
        queries
    )
    .lower()
)


assert (
    "trial"
    in
    query_text
)


assert (
    "prominent investor"
    in
    query_text
)


assert (
    "financial discrepancy"
    in
    query_text
)


assert (
    "fraud"
    in
    query_text
)


# ============================================================
# Multi-query retrieval
# ============================================================

retriever = (
    MultiQueryRetriever(

        hybrid_retriever=(
            FakeHybridRetriever()
        ),

        decomposer=(
            decomposer
        )
    )
)


results = (
    retriever.search(

        query,

        top_k=20
    )
)


documents = {

    item[
        "document_id"
    ]

    for item in results
}


print(
    "\n===== MULTI-QUERY DOCUMENTS ====="
)


print(
    sorted(
        documents
    )
)


# ============================================================
# Retrieval assertions
# ============================================================

assert (
    "doc_trial"
    in
    documents
)


assert (
    "doc_financial"
    in
    documents
)


assert (
    "doc_fraud"
    in
    documents
)


assert (
    "doc_investor"
    in
    documents
)


print(
    "\nMULTI-QUERY RETRIEVAL: OK"
)