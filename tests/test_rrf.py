import pytest

from adaptive_agentic_rag.retrieval.rrf import (
    reciprocal_rank_fusion,
)


def test_rrf_combines_rankings_and_preserves_score():

    dense = [
        {
            "id": "A",
            "score": 0.9,
        },
        {
            "id": "B",
            "score": 0.8,
        },
        {
            "id": "C",
            "score": 0.7,
        },
    ]


    bm25 = [
        {
            "id": "C",
            "score": 20.0,
        },
        {
            "id": "A",
            "score": 15.0,
        },
        {
            "id": "D",
            "score": 10.0,
        },
    ]


    results = (
        reciprocal_rank_fusion(
            [
                dense,
                bm25,
            ],
            top_k=4,
            k=60,
        )
    )


    assert [
        result[
            "id"
        ]
        for result
        in results
    ] == [
        "A",
        "C",
        "B",
        "D",
    ]


    for result in results:

        assert (
            result[
                "score"
            ]
            ==
            result[
                "rrf_score"
            ]
        )


    expected_a = (
        1 / 61
        +
        1 / 62
    )


    assert (
        results[0][
            "rrf_score"
        ]
        ==
        pytest.approx(
            expected_a
        )
    )


def test_rrf_preserves_vector_from_any_source():

    dense = [
        {
            "id": "A",
            "score": 0.9,
        }
    ]


    second_source = [
        {
            "id": "A",
            "score": 10.0,
            "vector": [
                1.0,
                2.0,
            ],
        }
    ]


    results = (
        reciprocal_rank_fusion(
            [
                dense,
                second_source,
            ]
        )
    )


    assert len(
        results
    ) == 1


    assert results[0][
        "vector"
    ] == [
        1.0,
        2.0,
    ]


def test_rrf_empty_top_k():

    assert (
        reciprocal_rank_fusion(
            [],
            top_k=0,
        )
        ==
        []
    )