import inspect
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


class FakeMultiQuery:

    def __init__(
        self,
        candidates,
    ):

        self.candidates = (
            candidates
        )


    def search(
        self,
        query,
        top_k,
    ):

        return (
            self.candidates[
                :top_k
            ]
        )


class FakeReranker:

    def rerank(
        self,
        query,
        candidates,
        top_k,
    ):

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
            ] = (
                10.0
                -
                index
            )


            output.append(
                item
            )


        return output


def build_fake_retriever():

    retriever = (
        RerankedRetriever.__new__(
            RerankedRetriever
        )
    )


    dense = SimpleNamespace(
        embedder=
            FakeEmbedder()
    )


    hybrid = SimpleNamespace(
        final_top_k=20,
        dense=dense,
        close=lambda: None,
    )


    candidates = [

        {
            "id":
                f"chunk_{index}",

            "document_id":
                f"doc_{index}",

            "text":
                f"document {index}",

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

            "score":
                0.1,
        }

        for index
        in range(
            20
        )
    ]


    retriever.hybrid = (
        hybrid
    )

    retriever.multi_query = (
        FakeMultiQuery(
            candidates
        )
    )

    retriever.reranker = (
        FakeReranker()
    )

    retriever.rerank_top_k = 10

    retriever.final_top_k = 5

    retriever.mmr_lambda = 0.7

    retriever._closed = False


    return retriever


def test_dense_retriever_argument_is_optional():

    signature = (
        inspect.signature(
            RerankedRetriever.__init__
        )
    )


    parameter = (
        signature.parameters[
            "dense_retriever"
        ]
    )


    assert (
        parameter.default
        is None
    )


def test_final_score_comes_from_cross_encoder(
    monkeypatch,
):

    retriever = (
        build_fake_retriever()
    )


    def fake_mmr_select(
        *,
        query_embedding,
        document_embeddings,
        documents,
        top_k,
        lambda_param,
    ):

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
            top_k=3,
        )
    )


    assert len(
        results
    ) == 3


    for result in results:

        assert (
            result[
                "score"
            ]
            ==
            result[
                "rerank_score"
            ]
        )


def test_empty_query_short_circuits():

    retriever = (
        build_fake_retriever()
    )


    assert (
        retriever.search(
            "",
            top_k=5,
        )
        ==
        []
    )