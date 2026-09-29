from types import SimpleNamespace

import numpy as np

import adaptive_agentic_rag.retrieval.reranked_retriever as reranked_module

from adaptive_agentic_rag.retrieval.reranked_retriever import (
    RerankedRetriever,
)


class RecordingEmbedder:

    def __init__(
        self,
    ):

        self.document_calls = []

        self.query_calls = []


    def encode_documents(
        self,
        texts,
    ):

        self.document_calls.append(
            list(
                texts
            )
        )


        return np.asarray(
            [
                [
                    0.5,
                    0.5,
                ]

                for _ in texts
            ],
            dtype=float,
        )


    def encode_queries(
        self,
        queries,
    ):

        self.query_calls.append(
            list(
                queries
            )
        )


        return np.asarray(
            [
                [
                    1.0,
                    0.0,
                ]
            ],
            dtype=float,
        )


class FakeMultiQuery:

    def search(
        self,
        query,
        top_k,
    ):

        return [
            {
                "id":
                    "chunk_a",

                "document_id":
                    "doc_a",

                "text":
                    "alpha evidence",

                "vector":
                    np.asarray(
                        [
                            1.0,
                            0.0,
                        ],
                        dtype=float,
                    ),
            },

            {
                "id":
                    "chunk_b",

                "document_id":
                    "doc_b",

                "text":
                    "beta evidence",
            },
        ]


class FakeReranker:

    def rerank(
        self,
        query,
        candidates,
        top_k,
    ):

        output = []


        for index, candidate in enumerate(
            candidates
        ):

            item = (
                candidate.copy()
            )


            item[
                "rerank_score"
            ] = float(
                2
                -
                index
            )


            output.append(
                item
            )


        return output[
            :top_k
        ]


def test_missing_vector_is_embedded_before_mmr(
    monkeypatch,
):

    embedder = (
        RecordingEmbedder()
    )


    retriever = (
        RerankedRetriever.__new__(
            RerankedRetriever
        )
    )


    retriever.hybrid = (
        SimpleNamespace(
            final_top_k=20,
            dense=
                SimpleNamespace(
                    embedder=
                        embedder
                ),
            close=lambda: None,
        )
    )


    retriever.multi_query = (
        FakeMultiQuery()
    )

    retriever.reranker = (
        FakeReranker()
    )

    retriever.rerank_top_k = 10

    retriever.final_top_k = 5

    retriever.mmr_lambda = 0.7

    retriever._closed = False


    captured = {}


    def fake_mmr_select(
        *,
        query_embedding,
        document_embeddings,
        documents,
        top_k,
        lambda_param,
    ):

        captured[
            "query_embedding"
        ] = query_embedding

        captured[
            "document_embeddings"
        ] = (
            document_embeddings
        )

        captured[
            "documents"
        ] = documents


        return (
            documents[
                :top_k
            ]
        )


    monkeypatch.setattr(
        reranked_module,
        "mmr_select",
        fake_mmr_select,
    )


    results = (
        retriever.search(
            "test query",
            top_k=2,
        )
    )


    assert len(
        results
    ) == 2


    assert (
        embedder.document_calls
        ==
        [
            [
                "beta evidence"
            ]
        ]
    )


    assert (
        embedder.query_calls
        ==
        [
            [
                "test query"
            ]
        ]
    )


    assert all(

        "vector"
        in document

        for document
        in captured[
            "documents"
        ]
    )


    assert len(
        captured[
            "document_embeddings"
        ]
    ) == 2


    assert all(

        result[
            "score"
        ]
        ==
        result[
            "rerank_score"
        ]

        for result
        in results
    )