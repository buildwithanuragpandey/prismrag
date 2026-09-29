from types import SimpleNamespace

from adaptive_agentic_rag.generation.relevance_filter import (
    ClaimRelevanceFilter,
)


class FakeReranker:

    def __init__(
        self,
        scores,
    ):

        self.scores = (
            scores
        )


    def rerank(
        self,
        query,
        documents,
        top_k=5,
        batch_size=8,
    ):

        output = []


        for document in (
            documents
        ):

            item = (
                document.copy()
            )


            item[
                "rerank_score"
            ] = float(
                self.scores[
                    document[
                        "text"
                    ]
                ]
            )


            output.append(
                item
            )


        output.sort(
            key=lambda item:
                item[
                    "rerank_score"
                ],
            reverse=True,
        )


        return output[
            :top_k
        ]


def claim(
    text,
    citation_id,
    *,
    supported=True,
):

    return SimpleNamespace(
        claim=
            text,

        citation_id=
            citation_id,

        supported=
            supported,
    )


def grounded(
    claims,
):

    return SimpleNamespace(
        claims=
            claims,
    )


def context_item(
    citation_id,
    source,
):

    return SimpleNamespace(
        citation_id=
            citation_id,

        source=
            source,
    )


def context(
    items,
):

    return SimpleNamespace(
        items=
            items,
    )


def test_three_source_question_expands_budget_to_three():

    mashable = (
        "Mashable reported continued and "
        "new Cyber Monday deals."
    )

    smh = (
        "The Sydney Morning Herald reported "
        "the antitrust lawsuit's effect on "
        "Amazon's stock."
    )

    cnbc = (
        "CNBC discussed the opportunity "
        "of selling on Amazon."
    )


    relevance_filter = (
        ClaimRelevanceFilter(
            reranker=FakeReranker(
                {
                    mashable:
                        10.0,

                    smh:
                        9.0,

                    cnbc:
                        8.0,
                }
            ),
            max_relevant_claims=2,
            max_adaptive_claims=4,
        )
    )


    result = relevance_filter.filter(
        query=(
            "Does the Mashable article discuss "
            "Cyber Monday deals, while The Sydney "
            "Morning Herald discusses Amazon's "
            "stock price, and CNBC discusses "
            "selling on Amazon?"
        ),

        grounded_claims=grounded(
            [
                claim(
                    mashable,
                    1,
                ),
                claim(
                    smh,
                    2,
                ),
                claim(
                    cnbc,
                    3,
                ),
            ]
        ),

        context=context(
            [
                context_item(
                    1,
                    "Mashable",
                ),
                context_item(
                    2,
                    "The Sydney Morning Herald",
                ),
                context_item(
                    3,
                    "CNBC | World Business News Leader",
                ),
            ]
        ),
    )


    assert (
        result.selection_mode
        ==
        "source_aware_adaptive"
    )


    assert (
        result.adaptive_budget
        ==
        3
    )


    assert len(
        result.relevant_claims
    ) == 3


    assert set(
        result.covered_sources
    ) == {
        "mashable",
        "the sydney morning herald",
        "cnbc",
    }


def test_best_claim_per_required_source_is_preserved():

    source_a_high = (
        "Source A high relevance."
    )

    source_a_second = (
        "Source A second relevance."
    )

    source_b = (
        "Source B relevant evidence."
    )


    relevance_filter = (
        ClaimRelevanceFilter(
            reranker=FakeReranker(
                {
                    source_a_high:
                        10.0,

                    source_a_second:
                        9.0,

                    source_b:
                        1.0,
                }
            ),
            max_relevant_claims=2,
        )
    )


    result = relevance_filter.filter(
        query=(
            "Does the Wired article agree "
            "with the TechCrunch article?"
        ),

        grounded_claims=grounded(
            [
                claim(
                    source_a_high,
                    1,
                ),
                claim(
                    source_a_second,
                    1,
                ),
                claim(
                    source_b,
                    2,
                ),
            ]
        ),

        context=context(
            [
                context_item(
                    1,
                    "Wired",
                ),
                context_item(
                    2,
                    "TechCrunch",
                ),
            ]
        ),
    )


    selected_texts = {
        item.claim

        for item
        in result.relevant_claims
    }


    assert (
        source_a_high
        in
        selected_texts
    )


    assert (
        source_b
        in
        selected_texts
    )


def test_source_aware_selection_never_rescues_below_floor_claim():

    source_a = (
        "Strong Wired evidence."
    )

    source_b_bad = (
        "Catastrophically irrelevant "
        "TechCrunch evidence."
    )


    relevance_filter = (
        ClaimRelevanceFilter(
            reranker=FakeReranker(
                {
                    source_a:
                        8.0,

                    source_b_bad:
                        -10.0,
                }
            )
        )
    )


    result = relevance_filter.filter(
        query=(
            "Does the Wired article agree "
            "with the TechCrunch article?"
        ),

        grounded_claims=grounded(
            [
                claim(
                    source_a,
                    1,
                ),
                claim(
                    source_b_bad,
                    2,
                ),
            ]
        ),

        context=context(
            [
                context_item(
                    1,
                    "Wired",
                ),
                context_item(
                    2,
                    "TechCrunch",
                ),
            ]
        ),
    )


    selected_texts = {
        item.claim

        for item
        in result.relevant_claims
    }


    filtered_texts = {
        item.claim

        for item
        in result.filtered_claims
    }


    assert (
        source_a
        in
        selected_texts
    )


    assert (
        source_b_bad
        not in
        selected_texts
    )


    assert (
        source_b_bad
        in
        filtered_texts
    )


def test_single_source_preserves_global_top_k_behavior():

    first = (
        "First strong claim."
    )

    second = (
        "Second strong claim."
    )

    third = (
        "Third claim."
    )


    relevance_filter = (
        ClaimRelevanceFilter(
            reranker=FakeReranker(
                {
                    first:
                        10.0,

                    second:
                        9.0,

                    third:
                        8.0,
                }
            ),
            max_relevant_claims=2,
        )
    )


    result = relevance_filter.filter(
        query=(
            "What does TechCrunch report "
            "about the company?"
        ),

        grounded_claims=grounded(
            [
                claim(
                    first,
                    1,
                ),
                claim(
                    second,
                    1,
                ),
                claim(
                    third,
                    1,
                ),
            ]
        ),

        context=context(
            [
                context_item(
                    1,
                    "TechCrunch",
                ),
            ]
        ),
    )


    assert (
        result.selection_mode
        ==
        "global_top_k"
    )


    assert [
        item.claim

        for item
        in result.relevant_claims
    ] == [
        first,
        second,
    ]


def test_sportingnews_alias_matches_sporting_news():

    first = (
        "Sporting News reported the Rangers lead."
    )

    second = (
        "The Roar reported the JackJumpers lead."
    )


    relevance_filter = (
        ClaimRelevanceFilter(
            reranker=FakeReranker(
                {
                    first:
                        5.0,

                    second:
                        4.0,
                }
            )
        )
    )


    result = relevance_filter.filter(
        query=(
            "Between The Roar and SportingNews, "
            "which report described the larger lead?"
        ),

        grounded_claims=grounded(
            [
                claim(
                    first,
                    2,
                ),
                claim(
                    second,
                    1,
                ),
            ]
        ),

        context=context(
            [
                context_item(
                    1,
                    "The Roar | Sports Writers Blog",
                ),
                context_item(
                    2,
                    "Sporting News",
                ),
            ]
        ),
    )


    assert (
        result.selection_mode
        ==
        "source_aware_adaptive"
    )


    assert set(
        result.required_sources
    ) == {
        "the roar",
        "sporting news",
    }