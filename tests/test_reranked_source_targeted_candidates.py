from types import SimpleNamespace

from adaptive_agentic_rag.retrieval.reranked_retriever import (
    RerankedRetriever,
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
                    "normal_1",

                "document_id":
                    "doc_normal",

                "text":
                    "normal candidate",

                "metadata": {
                    "source":
                        "TechCrunch",
                },
            },
        ]


class FakeBM25:

    def search_by_sources(
        self,
        query,
        sources,
        top_k_per_source,
    ):

        return [
            {
                "id":
                    "target_1",

                "document_id":
                    "doc_target",

                "text":
                    "target source candidate",

                "metadata": {
                    "source":
                        "The Age",
                },

                "source_targeted":
                    True,

                "source_target":
                    "The Age",
            },
        ]


def make_retriever():

    retriever = (
        object.__new__(
            RerankedRetriever
        )
    )


    retriever.multi_query = (
        FakeMultiQuery()
    )


    retriever.hybrid = (
        SimpleNamespace(
            bm25=
                FakeBM25()
        )
    )


    retriever.source_target_top_k = (
        20
    )


    return retriever


def test_target_source_candidate_is_injected():

    retriever = (
        make_retriever()
    )


    candidates = (
        retriever._collect_candidates(
            query=
                "The Age Google market practices",

            top_k=
                40,

            target_sources=[
                "The Age",
            ],
        )
    )


    ids = {
        item[
            "id"
        ]

        for item
        in candidates
    }


    assert (
        ids
        ==
        {
            "normal_1",
            "target_1",
        }
    )


def test_normal_candidate_generation_is_unchanged_without_targets():

    retriever = (
        make_retriever()
    )


    candidates = (
        retriever._collect_candidates(
            query=
                "ordinary query",

            top_k=
                40,

            target_sources=[],
        )
    )


    assert (
        [
            item[
                "id"
            ]

            for item
            in candidates
        ]
        ==
        [
            "normal_1",
        ]
    )