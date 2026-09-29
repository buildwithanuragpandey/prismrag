from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever,
)

from adaptive_agentic_rag.retrieval.hybrid_retriever import (
    HybridRetriever,
)

from adaptive_agentic_rag.retrieval.reranker import (
    BGEReranker,
)

from adaptive_agentic_rag.retrieval.mmr import (
    mmr_select,
)

from adaptive_agentic_rag.retrieval.multi_query_retriever import (
    MultiQueryRetriever,
)


DEFAULT_BM25_CORPUS_PATH = "data/processed/processed_corpus_v2.json"


class RerankedRetriever:

    def __init__(
        self,
        dense_retriever=None,
        hybrid_top_k: int = 20,
        rerank_top_k: int = 10,
        final_top_k: int = 5,
        mmr_lambda: float = 0.7,
        bm25_corpus_path: str = DEFAULT_BM25_CORPUS_PATH,
        source_target_top_k: int = 20,
    ):

        if dense_retriever is None:

            dense_retriever = (
                DenseRetriever()
            )


        self.hybrid = (
            HybridRetriever(
                dense_retriever=
                    dense_retriever,

                final_top_k=
                    hybrid_top_k,

                bm25_corpus_path=
                    bm25_corpus_path,
            )
        )


        self.multi_query = (
            MultiQueryRetriever(
                hybrid_retriever=
                    self.hybrid
            )
        )


        self.reranker = (
            BGEReranker()
        )


        self.rerank_top_k = (
            rerank_top_k
        )

        self.final_top_k = (
            final_top_k
        )

        self.mmr_lambda = (
            mmr_lambda
        )

        self.bm25_corpus_path = (
            bm25_corpus_path
        )

        self.source_target_top_k = (
            source_target_top_k
        )


        self._closed = False


    # ========================================================
    # Stable candidate identity
    # ========================================================

    @staticmethod
    def _candidate_key(
        item: dict,
    ):

        chunk_id = (
            item.get(
                "id"
            )
        )


        if chunk_id:

            return str(
                chunk_id
            )


        return (
            str(
                item.get(
                    "document_id",
                    "",
                )
            ),
            str(
                item.get(
                    "text",
                    "",
                )
            )[
                :200
            ],
        )


    # ========================================================
    # Merge normal + source-targeted candidate pools
    # ========================================================

    @classmethod
    def _merge_candidates(
        cls,
        normal_candidates: list[dict],
        targeted_candidates: list[dict],
    ) -> list[dict]:

        output = []

        by_key = {}


        for candidate in (
            list(
                normal_candidates
                or []
            )
            +
            list(
                targeted_candidates
                or []
            )
        ):

            key = (
                cls._candidate_key(
                    candidate
                )
            )


            existing = (
                by_key.get(
                    key
                )
            )


            if existing is None:

                copy = dict(
                    candidate
                )


                by_key[
                    key
                ] = (
                    copy
                )


                output.append(
                    copy
                )


                continue


            # ------------------------------------------------
            # Preserve vector from whichever path has one.
            # ------------------------------------------------

            if (
                existing.get(
                    "vector"
                )
                is None
                and
                candidate.get(
                    "vector"
                )
                is not None
            ):

                existing[
                    "vector"
                ] = (
                    candidate[
                        "vector"
                    ]
                )


            # ------------------------------------------------
            # Preserve source-target provenance.
            # ------------------------------------------------

            if candidate.get(
                "source_targeted"
            ):

                existing[
                    "source_targeted"
                ] = True


                existing[
                    "source_target"
                ] = (
                    candidate.get(
                        "source_target"
                    )
                )


        return output


    # ========================================================
    # Candidate generation
    #
    # Normal:
    #
    # MultiQuery Hybrid
    #
    # Retry:
    #
    # MultiQuery Hybrid
    #       +
    # source-constrained BM25
    #
    # Both feed the SAME reranker.
    # ========================================================

    def _collect_candidates(
        self,
        *,
        query: str,
        top_k: int,
        target_sources: list[str] | None,
    ) -> list[dict]:

        normal_candidates = (
            self.multi_query.search(
                query,
                top_k=
                    top_k,
            )
        )


        if not target_sources:

            return normal_candidates


        targeted_candidates = (
            self.hybrid
            .bm25
            .search_by_sources(
                query=
                    query,

                sources=
                    list(
                        target_sources
                    ),

                top_k_per_source=
                    self.source_target_top_k,
            )
        )


        return self._merge_candidates(
            normal_candidates,
            targeted_candidates,
        )


    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int | None = None,
        target_sources: list[str] | None = None,
    ) -> list[dict]:

        if top_k is None:

            top_k = (
                self.final_top_k
            )


        if (
            not query
            or
            top_k <= 0
        ):

            return []


        # ====================================================
        # Candidate budgets
        # ====================================================

        rerank_candidate_k = max(
            self.rerank_top_k,
            top_k * 2,
        )


        hybrid_candidate_k = max(
            self.hybrid.final_top_k,
            rerank_candidate_k,
        )


        multi_query_candidate_k = max(
            hybrid_candidate_k * 2,
            40,
        )


        # ====================================================
        # 1. Candidate generation
        #
        # target_sources=None:
        #     exact previous production behavior
        #
        # target_sources=[...]:
        #     normal pool + source-targeted BM25 injection
        # ====================================================

        candidates = (
            self._collect_candidates(
                query=
                    query,

                top_k=
                    multi_query_candidate_k,

                target_sources=
                    target_sources,
            )
        )


        if not candidates:

            return []


        # ====================================================
        # 2. ONE cross-encoder rerank
        #
        # Same behavior for normal and targeted retrieval.
        # ====================================================

        reranked = (
            self.reranker.rerank(
                query,
                candidates,
                top_k=
                    rerank_candidate_k,
            )
        )


        if not reranked:

            return []


        # ====================================================
        # 3. Ensure Dense vector
        # ====================================================

        valid_documents = []

        document_embeddings = []


        for item in reranked:

            vector = (
                item.get(
                    "vector"
                )
            )


            if vector is None:

                vector = (
                    self.hybrid
                    .dense
                    .embedder
                    .encode_documents(
                        [
                            item[
                                "text"
                            ]
                        ]
                    )[
                        0
                    ]
                )


                item[
                    "vector"
                ] = (
                    vector
                )


            valid_documents.append(
                item
            )


            document_embeddings.append(
                vector
            )


        reranked = (
            valid_documents
        )


        if not reranked:

            return []


        # ====================================================
        # 4. Query embedding for MMR
        # ====================================================

        query_embedding = (
            self.hybrid
            .dense
            .embedder
            .encode_queries(
                [
                    query
                ]
            )[
                0
            ]
        )


        # ====================================================
        # 5. Same MMR
        # ====================================================

        selected = (
            mmr_select(
                query_embedding=
                    query_embedding,

                document_embeddings=
                    document_embeddings,

                documents=
                    reranked,

                top_k=
                    top_k,

                lambda_param=
                    self.mmr_lambda,
            )
        )


        # ====================================================
        # 6. Same public scoring contract
        # ====================================================

        for item in selected:

            rerank_score = (
                item.get(
                    "rerank_score"
                )
            )


            if rerank_score is not None:

                item[
                    "score"
                ] = (
                    rerank_score
                )


        return selected


    # ========================================================
    # Cleanup
    # ========================================================

    def close(
        self,
    ):

        if self._closed:

            return


        self.hybrid.close()

        self._closed = True