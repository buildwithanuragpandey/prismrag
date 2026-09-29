from types import SimpleNamespace

from adaptive_agentic_rag.generation.answer_consistency import (
    AnswerConsistencyGuard,
)

from adaptive_agentic_rag.generation.generator import (
    GroundedGenerator,
)


def claim(
    text,
    citation_id,
):

    return SimpleNamespace(
        claim=text,
        citation_id=citation_id,
        relevance_score=5.0,
    )


def test_facts_first_output_is_parsed():

    raw = """
FACTS:
- Google was accused of anticompetitive practices.
- A second report discussed Google's market behavior.

DIRECT_ANSWER: Yes
""".strip()


    parsed = (
        GroundedGenerator
        ._parse_draft(
            raw
        )
    )


    assert (
        parsed.direct_answer
        ==
        "Yes"
    )


    assert (
        len(
            parsed.evidence_facts
        )
        ==
        2
    )


def test_yes_no_question_requires_yes_or_no():

    guard = (
        AnswerConsistencyGuard()
    )


    result = guard.validate(
        query=(
            "Does the first article agree "
            "with the second article?"
        ),

        direct_answer=
            "They appear similar.",

        relevant_claims=[
            claim(
                "The first article reports the same trend.",
                1,
            ),
            claim(
                "The second article reports the same trend.",
                2,
            ),
        ],
    )


    assert (
        result.valid
        is False
    )


def test_comparison_requires_multiple_citations():

    guard = (
        AnswerConsistencyGuard()
    )


    result = guard.validate(
        query=(
            "Does the Fortune article discuss FTX "
            "while the TechCrunch article discusses "
            "criminal charges?"
        ),

        direct_answer=
            "Yes",

        relevant_claims=[
            claim(
                "Gary Wang admitted guilt.",
                2,
            ),
            claim(
                "Caroline Ellison admitted guilt.",
                2,
            ),
        ],
    )


    assert (
        result.valid
        is False
    )


def test_comparison_accepts_multiple_grounded_citations():

    guard = (
        AnswerConsistencyGuard()
    )


    result = guard.validate(
        query=(
            "Does the Mashable article discuss deals "
            "while the Sydney Morning Herald article "
            "discusses an antitrust lawsuit?"
        ),

        direct_answer=
            "Yes",

        relevant_claims=[
            claim(
                "Mashable discussed Cyber Monday deals.",
                1,
            ),
            claim(
                "The Sydney Morning Herald discussed "
                "an antitrust lawsuit.",
                10,
            ),
        ],
    )


    assert (
        result.valid
        is True
    )


def test_entity_answer_must_appear_in_verified_claim():

    guard = (
        AnswerConsistencyGuard()
    )


    result = guard.validate(
        query=(
            "Which leading AI development company "
            "is behind ChatGPT?"
        ),

        direct_answer=
            "OpenAI",

        relevant_claims=[
            claim(
                "OpenAI is known as the company "
                "behind ChatGPT.",
                5,
            ),
        ],
    )


    assert (
        result.valid
        is True
    )


def test_publisher_is_rejected_as_organization_answer():

    guard = (
        AnswerConsistencyGuard()
    )


    result = guard.validate(
        query=(
            "Which organization, discussed in articles "
            "from The Roar Sports Writers Blog, is being "
            "encouraged to restore funding?"
        ),

        direct_answer=
            "The Roar Sports Writers Blog",

        relevant_claims=[
            claim(
                "The Roar Sports Writers Blog mentions "
                "that Rugby Australia should restore funding.",
                1,
            ),
        ],
    )


    assert (
        result.valid
        is False
    )


def test_news_source_is_allowed_when_question_asks_for_source():

    guard = (
        AnswerConsistencyGuard()
    )


    result = guard.validate(
        query=(
            "Between two match reports, which news source "
            "reported the larger lead?"
        ),

        direct_answer=
            "The Roar Sports Writers Blog",

        relevant_claims=[
            claim(
                "The Roar Sports Writers Blog reported "
                "the larger lead.",
                1,
            ),
            claim(
                "Sporting News reported another lead.",
                2,
            ),
        ],
    )


    assert (
        result.valid
        is True
    )