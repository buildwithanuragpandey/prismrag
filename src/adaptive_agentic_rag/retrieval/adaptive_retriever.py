from adaptive_agentic_rag.agents.query_router import (
    QueryRouter,
)

from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever,
)

from adaptive_agentic_rag.retrieval.reranked_retriever import (
    RerankedRetriever,
)


DEFAULT_DENSE_COLLECTION = (
    "multihop_chunks_v2"
)

DEFAULT_BM25_CORPUS_PATH = (
    "data/processed/"
    "processed_corpus_v2.json"
)


class AdaptiveRetriever:

    def __init__(
        self,
        collection_name: str = (
            DEFAULT_DENSE_COLLECTION
        ),
        bm25_corpus_path: str = (
            DEFAULT_BM25_CORPUS_PATH
        ),
    ):

        self.collection_name = (
            collection_name
        )

        self.bm25_corpus_path = (
            bm25_corpus_path
        )


        self.router = (
            QueryRouter()
        )


        self.dense = (
            DenseRetriever(
                collection_name=
                    collection_name
            )
        )


        self.reranked = (
            RerankedRetriever(
                dense_retriever=
                    self.dense,

                bm25_corpus_path=
                    bm25_corpus_path,
            )
        )


        self._closed = False


    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        target_sources: list[str] | None = None,
    ) -> dict:

        if (
            not query
            or
            top_k <= 0
        ):

            return {
                "decision":
                    None,

                "results":
                    [],
            }


        decision = (
            self.router.route(
                query
            )
        )


        target_sources = [
            str(
                source
            ).strip()

            for source in (
                target_sources
                or []
            )

            if str(
                source
            ).strip()
        ]


        # ====================================================
        # Structurally approved source-targeted retry
        #
        # Always uses the heavy retrieval path because the
        # source-targeted candidate injection occurs before
        # the shared cross-encoder reranker.
        #
        # Normal routing remains unchanged when the list is
        # empty.
        # ====================================================

        if target_sources:

            results = (
                self.reranked.search(
                    query,
                    top_k=
                        top_k,

                    target_sources=
                        target_sources,
                )
            )


        # ====================================================
        # Normal simple query
        # ====================================================

        elif (
            decision[
                "retrieval_strategy"
            ]
            ==
            "dense"
        ):

            results = (
                self.dense.search(
                    query,
                    top_k=
                        top_k,
                )
            )


        # ====================================================
        # Normal hard query
        # ====================================================

        else:

            results = (
                self.reranked.search(
                    query,
                    top_k=
                        top_k,
                )
            )


        return {
            "decision":
                decision,

            "results":
                results,
        }


    # ========================================================
    # Cleanup
    # ========================================================

    def close(
        self,
    ):

        if self._closed:

            return


        self.reranked.close()

        self._closed = True