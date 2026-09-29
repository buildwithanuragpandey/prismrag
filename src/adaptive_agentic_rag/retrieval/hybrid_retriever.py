from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever,
)

from adaptive_agentic_rag.retrieval.bm25_retriever import (
    BM25Retriever,
)

from adaptive_agentic_rag.retrieval.rrf import (
    reciprocal_rank_fusion,
)


DEFAULT_BM25_CORPUS_PATH = (
    "data/processed/processed_corpus_v2.json"
)


class HybridRetriever:

    def __init__(
        self,
        dense_retriever=None,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        final_top_k: int = 20,
        bm25_corpus_path: str = DEFAULT_BM25_CORPUS_PATH,
    ):

        self.dense = (
            dense_retriever
            if dense_retriever is not None
            else DenseRetriever()
        )

        self.bm25 = BM25Retriever(
            corpus_path=bm25_corpus_path
        )

        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k

        self.bm25_corpus_path = (
            bm25_corpus_path
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:

        if top_k is None:
            top_k = self.final_top_k

        if (
            not query
            or top_k <= 0
        ):
            return []

        # ====================================================
        # Candidate depths
        # ====================================================

        dense_candidate_k = max(
            self.dense_top_k,
            top_k,
        )

        bm25_candidate_k = max(
            self.bm25_top_k,
            top_k,
        )

        # ====================================================
        # Dense
        # ====================================================

        dense_results = self.dense.search(
            query,
            top_k=dense_candidate_k,
        )

        # ====================================================
        # BM25
        # ====================================================

        bm25_results = self.bm25.search(
            query,
            top_k=bm25_candidate_k,
        )

        # ====================================================
        # RRF
        # ====================================================

        fused_results = (
            reciprocal_rank_fusion(
                [
                    dense_results,
                    bm25_results,
                ],
                top_k=top_k,
            )
        )

        return fused_results

    def close(self):

        self.dense.close()