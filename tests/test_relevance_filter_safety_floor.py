from types import SimpleNamespace

from adaptive_agentic_rag.generation.relevance_filter import (
    ClaimRelevanceFilter,
    DEFAULT_MIN_RELEVANCE_SCORE,
)


class FakeReranker:

    def __init__(
        self,
        scores_by_text,
    ):

        self.scores_by_text = (
            scores_by_text
        )


    def rerank(
        self,
        query,
        documents,
        top_k=5,
        batch_size=8,
    ):

        ranked = []


        for document in documents:

            item = document.copy()

            item[
                "rerank_score"
            ] = float(
                self.scores_by_text[
                    document[
                        "text"
                    ]
                ]
            )


            ranked.append(
                item
            )


        ranked.sort(
            key=lambda item:
                item[
                    "rerank_score"
                ],
            reverse=True,
        )


        return ranked[
            :top_k
        ]


def make_claim(
    text,
    *,
    citation_id=1,
    supported=True,
):

    return SimpleNamespace(

        claim=text,

        citation_id=citation_id,

        supported=supported,
    )


def make_grounded(
    claims,
):

    return SimpleNamespace(
        claims=claims
    )


def test_calibrated_positive_is_preserved():

    claim_text = (
        "Altman's perceived lack of candor."
    )


    reranker = FakeReranker(
        {
            claim_text:
                -3.7592,
        }
    )


    relevance_filter = (
        ClaimRelevanceFilter(
            reranker=reranker,
        )
    )


    result = relevance_filter.filter(

        query=(
            "Did the later Fortune article "
            "contradict the TechCrunch report "
            "about Sam Altman's departure?"
        ),

        grounded_claims=make_grounded(
            [
                make_claim(
                    claim_text
                )
            ]
        ),
    )


    assert len(
        result.relevant_claims
    ) == 1


    assert len(
        result.filtered_claims
    ) == 0


    assert (
        result.relevant_claims[
            0
        ].relevance_score
        ==
        -3.7592
    )


def test_catastrophic_negative_is_filtered_even_when_only_claim():

    claim_text = (
        "France backs African mobile video "
        "startup StarNews Mobile in a "
        "$3 million round."
    )


    reranker = FakeReranker(
        {
            claim_text:
                -10.1626,
        }
    )


    relevance_filter = (
        ClaimRelevanceFilter(
            reranker=reranker,
        )
    )


    result = relevance_filter.filter(

        query=(
            "Which West African country "
            "was projected to grow its GDP "
            "and become self-sufficient "
            "in rice production?"
        ),

        grounded_claims=make_grounded(
            [
                make_claim(
                    claim_text
                )
            ]
        ),
    )


    assert (
        result.relevant_claims
        ==
        []
    )


    assert len(
        result.filtered_claims
    ) == 1


    assert (
        result.filtered_claims[
            0
        ].relevance_score
        ==
        -10.1626
    )


def test_floor_boundary_is_inclusive():

    claim_text = (
        "Boundary claim."
    )


    reranker = FakeReranker(
        {
            claim_text:
                DEFAULT_MIN_RELEVANCE_SCORE,
        }
    )


    relevance_filter = (
        ClaimRelevanceFilter(
            reranker=reranker,
        )
    )


    result = relevance_filter.filter(

        query="Boundary query",

        grounded_claims=make_grounded(
            [
                make_claim(
                    claim_text
                )
            ]
        ),
    )


    assert len(
        result.relevant_claims
    ) == 1


def test_top_k_is_applied_after_safety_floor():

    catastrophic = (
        "Catastrophic irrelevant claim."
    )

    good_one = (
        "First relevant claim."
    )

    good_two = (
        "Second relevant claim."
    )

    good_three = (
        "Third relevant claim."
    )


    reranker = FakeReranker(
        {
            catastrophic:
                -10.0,

            good_one:
                8.0,

            good_two:
                7.0,

            good_three:
                6.0,
        }
    )


    relevance_filter = (
        ClaimRelevanceFilter(

            reranker=reranker,

            max_relevant_claims=2,
        )
    )


    result = relevance_filter.filter(

        query="Relevant query",

        grounded_claims=make_grounded(
            [
                make_claim(
                    catastrophic,
                    citation_id=1,
                ),

                make_claim(
                    good_one,
                    citation_id=2,
                ),

                make_claim(
                    good_two,
                    citation_id=3,
                ),

                make_claim(
                    good_three,
                    citation_id=4,
                ),
            ]
        ),
    )


    assert [
        claim.claim

        for claim
        in result.relevant_claims
    ] == [
        good_one,
        good_two,
    ]


    filtered_texts = {
        claim.claim

        for claim
        in result.filtered_claims
    }


    assert catastrophic in (
        filtered_texts
    )


    assert good_three in (
        filtered_texts
    )


def test_placeholder_claim_is_filtered_before_reranking():

    placeholder = (
        "[TechCrunch article mentions "
        "Google's antitrust suit]"
    )


    class RerankerThatMustNotRun:

        def rerank(
            self,
            *args,
            **kwargs,
        ):

            raise AssertionError(
                "Malformed claim should not "
                "reach the reranker."
            )


    relevance_filter = (
        ClaimRelevanceFilter(
            reranker=(
                RerankerThatMustNotRun()
            )
        )
    )


    result = relevance_filter.filter(

        query="Google antitrust question",

        grounded_claims=make_grounded(
            [
                make_claim(
                    placeholder
                )
            ]
        ),
    )


    assert (
        result.relevant_claims
        ==
        []
    )


    assert len(
        result.filtered_claims
    ) == 1


    assert (
        result.total_claims
        ==
        1
    )


def test_unsupported_claims_do_not_enter_relevance_filter():

    supported_text = (
        "Supported useful claim."
    )


    reranker = FakeReranker(
        {
            supported_text:
                5.0,
        }
    )


    relevance_filter = (
        ClaimRelevanceFilter(
            reranker=reranker,
        )
    )


    result = relevance_filter.filter(

        query="Useful question",

        grounded_claims=make_grounded(
            [
                make_claim(
                    supported_text,
                    supported=True,
                ),

                make_claim(
                    "Unsupported hallucination.",
                    supported=False,
                ),
            ]
        ),
    )


    assert (
        result.total_claims
        ==
        1
    )


    assert len(
        result.relevant_claims
    ) == 1