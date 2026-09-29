from adaptive_agentic_rag.retrieval.adaptive_retriever import (
    AdaptiveRetriever,
)


class FakeRouter:

    def route(
        self,
        query,
    ):

        return {
            "query_type":
                "simple",

            "retrieval_strategy":
                "dense",

            "rerank":
                False,

            "mmr":
                False,
        }


class FakeDense:

    def __init__(
        self,
    ):

        self.called = False


    def search(
        self,
        query,
        top_k,
    ):

        self.called = True

        return [
            {
                "id":
                    "dense",
            }
        ]


class FakeReranked:

    def __init__(
        self,
    ):

        self.call = None


    def search(
        self,
        query,
        top_k,
        target_sources=None,
    ):

        self.call = {
            "query":
                query,

            "top_k":
                top_k,

            "target_sources":
                target_sources,
        }


        return [
            {
                "id":
                    "targeted",
            }
        ]


def test_target_sources_force_heavy_retry_path():

    retriever = (
        object.__new__(
            AdaptiveRetriever
        )
    )


    retriever.router = (
        FakeRouter()
    )


    retriever.dense = (
        FakeDense()
    )


    retriever.reranked = (
        FakeReranked()
    )


    retriever._closed = False


    result = (
        retriever.search(
            "The Age Google",
            top_k=10,
            target_sources=[
                "The Age",
            ],
        )
    )


    assert (
        retriever.dense.called
        is False
    )


    assert (
        retriever.reranked.call[
            "target_sources"
        ]
        ==
        [
            "The Age",
        ]
    )


    assert (
        result[
            "results"
        ][
            0
        ][
            "id"
        ]
        ==
        "targeted"
    )