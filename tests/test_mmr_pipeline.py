from types import SimpleNamespace

import numpy as np

import adaptive_agentic_rag.retrieval.reranked_retriever as reranked_module

from adaptive_agentic_rag.retrieval.reranked_retriever import (
    RerankedRetriever,
)


class FakeEmbedder:

    def encode_queries(
        self,
        queries,
    ):

        return np.asarray(
            [
                [
                    1.0,
                    0.0,
                ]
            ],
            dtype=float,
        )


    def encode_documents(
        self,
        texts,
    ):

        return np.asarray(
            [
                [
                    1.0,
                    0.0,
                ]

                for _ in texts
            ],
            dtype=float,
        )


class RecordingMultiQuery:

    def __init__(
        self,
        candidates,
    ):

        self.candidates = (
            candidates
        )

        self.last_top_k = None


    def search(
        self,
        query,
        top_k,
    ):

        self.last_top_k = (
            top_k
        )


        return (
            self.candidates[
                :top_k
            ]
        )


class RecordingReranker:

    def __init__(
        self,
    ):

        self.last_top_k = None


    def rerank(
        self,
        query,
        candidates,
        top_k,
    ):

        self.last_top_k = (
            top_k
        )


        output = []


        for index, candidate in enumerate(
            candidates[
                :top_k
            ]
        ):

            item = (
                candidate.copy()
            )


            item[
                "rerank_score"
            ] = float(
                100
                -
                index
            )


            output.append(
                item
            )


        return output


def test_candidate_budgets_and_mmr_flow(
    monkeypatch,
):

    candidates = [

        {
            "id":
                f"chunk_{index}",

            "document_id":
                f"doc_{index}",

            "text":
                f"text {index}",

            "vector":
                np.asarray(
                    [
                        1.0,
                        float(
                            index
                        ),
                    ],
                    dtype=float,
                ),
        }

        for index
        in range(
            40
        )
    ]


    multi_query = (
        RecordingMultiQuery(
            candidates
        )
    )


    reranker = (
        RecordingReranker()
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
                        FakeEmbedder()
                ),
            close=lambda: None,
        )
    )


    retriever.multi_query = (
        multi_query
    )

    retriever.reranker = (
        reranker
    )

    retriever.rerank_top_k = 10

    retriever.final_top_k = 5

    retriever.mmr_lambda = 0.7

    retriever._closed = False


    mmr_call = {}


    def fake_mmr_select(
        *,
        query_embedding,
        document_embeddings,
        documents,
        top_k,
        lambda_param,
    ):

        mmr_call[
            "candidate_count"
        ] = len(
            documents
        )

        mmr_call[
            "embedding_count"
        ] = len(
            document_embeddings
        )

        mmr_call[
            "top_k"
        ] = top_k

        mmr_call[
            "lambda"
        ] = lambda_param


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
            "best amazon cyber monday deals",
            top_k=5,
        )
    )


    assert (
        multi_query.last_top_k
        ==
        40
    )


    assert (
        reranker.last_top_k
        ==
        10
    )


    assert (
        mmr_call[
            "candidate_count"
        ]
        ==
        10
    )


    assert (
        mmr_call[
            "embedding_count"
        ]
        ==
        10
    )


    assert (
        mmr_call[
            "top_k"
        ]
        ==
        5
    )


    assert (
        mmr_call[
            "lambda"
        ]
        ==
        0.7
    )


    assert len(
        results
    ) == 5