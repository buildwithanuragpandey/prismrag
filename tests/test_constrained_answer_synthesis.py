from types import SimpleNamespace

from adaptive_agentic_rag.generation.constrained_answer_synthesis import (
    ConstrainedAnswerSynthesizer,
)


def claim(
    text,
    citation_id,
):

    return SimpleNamespace(
        claim=
            text,

        citation_id=
            citation_id,
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


class FakeBinarySynthesizer(
    ConstrainedAnswerSynthesizer
):

    def __init__(
        self,
        *,
        yes_score,
        no_score,
    ):

        self.fake_yes_score = (
            yes_score
        )

        self.fake_no_score = (
            no_score
        )


    def _score_yes_no_candidates(
        self,
        **kwargs,
    ):

        return {
            "yes":
                self.fake_yes_score,

            "no":
                self.fake_no_score,
        }


def test_yes_no_uses_constrained_score_not_draft_answer():

    synthesizer = (
        FakeBinarySynthesizer(
            yes_score=
                -0.5,

            no_score=
                -4.0,
        )
    )


    result = synthesizer.synthesize(
        query=(
            "Does the first article agree "
            "with the second article?"
        ),

        draft_direct_answer=
            "No",

        relevant_claims=[
            claim(
                "The first article reports "
                "the same pattern.",
                1,
            ),
            claim(
                "The second article reports "
                "the same pattern.",
                2,
            ),
        ],

        context=context(
            [
                context_item(
                    1,
                    "Source A",
                ),
                context_item(
                    2,
                    "Source B",
                ),
            ]
        ),

        model=None,
        tokenizer=None,
        device="cpu",
    )


    assert (
        result.accepted
        is True
    )


    assert (
        result.final_answer
        ==
        "Yes"
    )


    assert (
        result.yes_score
        >
        result.no_score
    )


def test_yes_no_can_select_no():

    synthesizer = (
        FakeBinarySynthesizer(
            yes_score=
                -5.0,

            no_score=
                -0.2,
        )
    )


    result = synthesizer.synthesize(
        query=(
            "Was the later report inconsistent "
            "with the earlier report?"
        ),

        draft_direct_answer=
            "Yes",

        relevant_claims=[
            claim(
                "The earlier report described "
                "the same relationship.",
                1,
            ),
            claim(
                "The later report described "
                "the same relationship.",
                2,
            ),
        ],

        context=context(
            [
                context_item(
                    1,
                    "Source A",
                ),
                context_item(
                    2,
                    "Source B",
                ),
            ]
        ),

        model=None,
        tokenizer=None,
        device="cpu",
    )


    assert (
        result.final_answer
        ==
        "No"
    )


def test_multi_evidence_yes_no_rejects_single_citation():

    synthesizer = (
        FakeBinarySynthesizer(
            yes_score=
                1.0,

            no_score=
                0.0,
        )
    )


    result = synthesizer.synthesize(
        query=(
            "Does the Fortune article discuss FTX "
            "while the TechCrunch article discusses "
            "criminal charges?"
        ),

        draft_direct_answer=
            "Yes",

        relevant_claims=[
            claim(
                "Gary Wang pleaded guilty.",
                2,
            ),
            claim(
                "Caroline Ellison pleaded guilty.",
                2,
            ),
        ],

        context=context(
            [
                context_item(
                    2,
                    "TechCrunch",
                ),
            ]
        ),

        model=None,
        tokenizer=None,
        device="cpu",
    )


    assert (
        result.accepted
        is False
    )


def test_named_sources_must_be_covered():

    synthesizer = (
        FakeBinarySynthesizer(
            yes_score=
                1.0,

            no_score=
                0.0,
        )
    )


    result = synthesizer.synthesize(
        query=(
            "Does the Mashable article discuss deals "
            "while The Sydney Morning Herald discusses "
            "Amazon's stock and CNBC discusses selling "
            "on Amazon?"
        ),

        draft_direct_answer=
            "No",

        relevant_claims=[
            claim(
                "Mashable discussed Cyber Monday deals.",
                1,
            ),
            claim(
                "The Sydney Morning Herald discussed "
                "Amazon's stock.",
                2,
            ),
        ],

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
                    "CNBC",
                ),
            ]
        ),

        model=None,
        tokenizer=None,
        device="cpu",
    )


    assert (
        result.accepted
        is False
    )


def test_verified_entity_is_accepted():

    synthesizer = (
        ConstrainedAnswerSynthesizer()
    )


    result = synthesizer.synthesize(
        query=(
            "Which leading AI development company "
            "is behind ChatGPT?"
        ),

        draft_direct_answer=
            "OpenAI",

        relevant_claims=[
            claim(
                "OpenAI is the company behind ChatGPT.",
                5,
            ),
        ],

        context=context(
            [
                context_item(
                    5,
                    "TechCrunch",
                ),
            ]
        ),

        model=None,
        tokenizer=None,
        device="cpu",
    )


    assert (
        result.accepted
        is True
    )


    assert (
        result.final_answer
        ==
        "OpenAI"
    )


def test_unverified_entity_is_rejected():

    synthesizer = (
        ConstrainedAnswerSynthesizer()
    )


    result = synthesizer.synthesize(
        query=(
            "Which leading AI development company "
            "is behind ChatGPT?"
        ),

        draft_direct_answer=
            "Google",

        relevant_claims=[
            claim(
                "OpenAI is the company behind ChatGPT.",
                5,
            ),
        ],

        context=context(
            [
                context_item(
                    5,
                    "TechCrunch",
                ),
            ]
        ),

        model=None,
        tokenizer=None,
        device="cpu",
    )


    assert (
        result.accepted
        is False
    )


def test_publisher_cannot_answer_organization_question():

    synthesizer = (
        ConstrainedAnswerSynthesizer()
    )


    result = synthesizer.synthesize(
        query=(
            "Which organization, discussed in articles "
            "from The Roar Sports Writers Blog, is being "
            "encouraged to restore funding?"
        ),

        draft_direct_answer=
            "The Roar Sports Writers Blog",

        relevant_claims=[
            claim(
                "The Roar Sports Writers Blog reports "
                "that Rugby Australia should restore funding.",
                1,
            ),
        ],

        context=context(
            [
                context_item(
                    1,
                    "The Roar Sports Writers Blog",
                ),
            ]
        ),

        model=None,
        tokenizer=None,
        device="cpu",
    )


    assert (
        result.accepted
        is False
    )


def test_source_answer_allowed_when_question_asks_for_source():

    synthesizer = (
        ConstrainedAnswerSynthesizer()
    )


    result = synthesizer.synthesize(
        query=(
            "Between the two reports, which news source "
            "reported the larger lead?"
        ),

        draft_direct_answer=
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

        context=context(
            [
                context_item(
                    1,
                    "The Roar Sports Writers Blog",
                ),
                context_item(
                    2,
                    "Sporting News",
                ),
            ]
        ),

        model=None,
        tokenizer=None,
        device="cpu",
    )


    assert (
        result.accepted
        is True
    )


    assert (
        result.final_answer
        ==
        "The Roar Sports Writers Blog"
    )


def test_other_answer_type_preserves_draft():

    synthesizer = (
        ConstrainedAnswerSynthesizer()
    )


    result = synthesizer.synthesize(
        query=(
            "Explain the reported change."
        ),

        draft_direct_answer=
            "The reported change was an increase.",

        relevant_claims=[
            claim(
                "The value increased.",
                1,
            ),
        ],

        context=context(
            [
                context_item(
                    1,
                    "Example",
                ),
            ]
        ),

        model=None,
        tokenizer=None,
        device="cpu",
    )


    assert (
        result.accepted
        is True
    )


    assert (
        result.final_answer
        ==
        "The reported change was an increase."
    )