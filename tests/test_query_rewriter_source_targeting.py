from adaptive_agentic_rag.agents.query_rewriter import (
    QueryRewriter,
)


def rewriter():

    return QueryRewriter()


def test_the_age_retry_becomes_source_targeted():

    query = (
        "Has the portrayal of Google's market practices "
        "in reports by The Age after October 22, 2023, "
        "remained consistent with the depiction in "
        "The Verge's coverage of the Epic v. Google case, "
        "and with TechCrunch's report on the class action "
        "antitrust suit filed against Google?"
    )


    rewritten = (
        rewriter().rewrite(
            query=
                query,

            query_type=
                "multihop",

            required_sources=[
                "The Age",
                "The Verge",
                "TechCrunch",
            ],

            covered_sources=[
                "The Verge",
                "TechCrunch",
            ],

            missing_sources=[
                "The Age",
            ],
        )
    )


    assert (
        rewritten.startswith(
            "The Age "
        )
    )


    lowered = (
        rewritten.lower()
    )


    assert (
        "google"
        in lowered
    )


    assert (
        "2023"
        in lowered
    )


    assert (
        "supporting evidence"
        not in lowered
    )


    assert (
        "techcrunch"
        not in lowered
    )


def test_cbssports_retry_preserves_late_query_entities():

    query = (
        "Which NFL team, featured in articles from both "
        "'Sporting News' and 'CBSSports.com', faced the "
        "potential of being closely followed in the wild "
        "card race by three other teams, might have opted "
        "for a field goal in a Monday Night Football game, "
        "recently changed their passing game strategy, and "
        "has seen comparable offensive production from "
        "Josh Dobbs and Kirk Cousins?"
    )


    rewritten = (
        rewriter().rewrite(
            query=
                query,

            query_type=
                "multihop",

            required_sources=[
                "Sporting News",
                "CBSSports.com",
            ],

            covered_sources=[
                "Sporting News",
            ],

            missing_sources=[
                "CBSSports.com",
            ],
        )
    )


    lowered = (
        rewritten.lower()
    )


    assert (
        rewritten.startswith(
            "CBSSports.com "
        )
    )


    assert (
        "josh"
        in lowered
    )


    assert (
        "dobbs"
        in lowered
    )


    assert (
        "kirk"
        in lowered
    )


    assert (
        "cousins"
        in lowered
    )


    assert (
        "sporting news"
        not in lowered
    )


def test_targeted_retry_removes_already_covered_source():

    rewritten = (
        rewriter().rewrite(
            query=(
                "The Verge discusses Apple while "
                "The New York Times discusses the "
                "latest iPad Pro performance upgrade."
            ),

            query_type=
                "multihop",

            required_sources=[
                "The Verge",
                "New York Times",
            ],

            covered_sources=[
                "The Verge",
            ],

            missing_sources=[
                "New York Times",
            ],
        )
    )


    lowered = (
        rewritten.lower()
    )


    assert (
        rewritten.startswith(
            "New York Times "
        )
    )


    assert (
        "the verge"
        not in lowered
    )


def test_multiple_missing_sources_are_preserved():

    rewritten = (
        rewriter().rewrite(
            query=(
                "Reuters discussed the launch and Bloomberg "
                "discussed the financial impact."
            ),

            query_type=
                "multihop",

            required_sources=[
                "Reuters",
                "Bloomberg",
            ],

            covered_sources=[],

            missing_sources=[
                "Reuters",
                "Bloomberg",
            ],
        )
    )


    assert (
        rewritten.startswith(
            "Reuters Bloomberg "
        )
    )


def test_no_missing_source_preserves_legacy_multihop_behavior():

    rewritten = (
        rewriter().rewrite(
            query=(
                "What evidence explains the change?"
            ),

            query_type=
                "multihop",

            missing_sources=[],
        )
    )


    assert (
        rewritten.endswith(
            "supporting evidence"
        )
    )


def test_simple_query_without_retry_telemetry_is_unchanged():

    query = (
        "What is the capital of France?"
    )


    rewritten = (
        rewriter().rewrite(
            query=
                query,

            query_type=
                "simple",
        )
    )


    assert (
        rewritten
        ==
        query
    )