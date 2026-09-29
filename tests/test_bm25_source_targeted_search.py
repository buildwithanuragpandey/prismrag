import json

from adaptive_agentic_rag.retrieval.bm25_retriever import (
    BM25Retriever,
)


def build_corpus(
    tmp_path,
):

    corpus = [
        {
            "id":
                "chunk_age_1",

            "document_id":
                "doc_age",

            "text":
                (
                    "Google market practices and "
                    "search advertising revenue."
                ),

            "metadata": {
                "source":
                    "The Age",

                "title":
                    "Google Search practices",
            },
        },

        {
            "id":
                "chunk_nyt_1",

            "document_id":
                "doc_nyt",

            "text":
                (
                    "The latest iPad Pro received "
                    "a performance upgrade."
                ),

            "metadata": {
                "source":
                    "The New York Times",

                "title":
                    "iPad Pro performance",
            },
        },

        {
            "id":
                "chunk_other",

            "document_id":
                "doc_other",

            "text":
                (
                    "Google market practices were "
                    "discussed elsewhere."
                ),

            "metadata": {
                "source":
                    "TechCrunch",

                "title":
                    "Google",
            },
        },
    ]


    path = (
        tmp_path
        /
        "corpus.json"
    )


    path.write_text(
        json.dumps(
            corpus
        ),
        encoding="utf-8",
    )


    return path


def test_source_targeted_search_returns_only_requested_source(
    tmp_path,
):

    retriever = (
        BM25Retriever(
            corpus_path=
                str(
                    build_corpus(
                        tmp_path
                    )
                )
        )
    )


    results = (
        retriever.search_by_sources(
            query=
                "Google market practices",

            sources=[
                "The Age",
            ],

            top_k_per_source=
                10,
        )
    )


    assert results


    assert all(
        (
            item[
                "metadata"
            ][
                "source"
            ]
            ==
            "The Age"
        )

        for item
        in results
    )


def test_source_alias_matches_the_new_york_times(
    tmp_path,
):

    retriever = (
        BM25Retriever(
            corpus_path=
                str(
                    build_corpus(
                        tmp_path
                    )
                )
        )
    )


    results = (
        retriever.search_by_sources(
            query=
                "iPad Pro performance",

            sources=[
                "New York Times",
            ],

            top_k_per_source=
                10,
        )
    )


    assert (
        len(
            results
        )
        ==
        1
    )


    assert (
        results[
            0
        ][
            "metadata"
        ][
            "source"
        ]
        ==
        "The New York Times"
    )