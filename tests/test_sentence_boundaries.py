from adaptive_agentic_rag.generation.context_builder import (
    ContextBuilder,
)

from adaptive_agentic_rag.generation.claim_grounder import (
    ClaimGrounder,
)

from adaptive_agentic_rag.generation.sentence_splitter import (
    split_sentences,
)


def test_shared_splitter_preserves_epic_v_google():

    text = (
        "Google faced allegations during the "
        "Epic v. Google trial. "
        "The case involved app-store practices."
    )


    sentences = (
        split_sentences(
            text
        )
    )


    assert sentences == [
        (
            "Google faced allegations during the "
            "Epic v. Google trial."
        ),
        (
            "The case involved app-store practices."
        ),
    ]


def test_shared_splitter_preserves_chiefs_vs_chargers():

    text = (
        "Swift attended the Chiefs vs. Chargers game. "
        "She attended other games as well."
    )


    sentences = (
        split_sentences(
            text
        )
    )


    assert sentences == [
        (
            "Swift attended the Chiefs vs. Chargers game."
        ),
        (
            "She attended other games as well."
        ),
    ]


def test_context_builder_preserves_v_abbreviation():

    builder = (
        ContextBuilder()
    )


    sentences = (
        builder._split_sentences(
            (
                "Google faced allegations during the "
                "Epic v. Google trial. "
                "Another sentence follows."
            )
        )
    )


    assert sentences == [
        (
            "Google faced allegations during the "
            "Epic v. Google trial."
        ),
        "Another sentence follows.",
    ]


def test_claim_grounder_preserves_v_abbreviation_without_loading_model():

    grounder = (
        ClaimGrounder.__new__(
            ClaimGrounder
        )
    )


    sentences = (
        grounder._split_sentences(
            (
                "Google faced allegations during the "
                "Epic v. Google trial. "
                "Another evidence sentence follows."
            )
        )
    )


    assert sentences == [
        (
            "Google faced allegations during the "
            "Epic v. Google trial."
        ),
        (
            "Another evidence sentence follows."
        ),
    ]


def test_context_builder_handles_bullet_newlines():

    builder = (
        ContextBuilder()
    )


    text = (
        "- First relevant statement.\n"
        "- Second relevant statement."
    )


    sentences = (
        builder._split_sentences(
            text
        )
    )


    assert sentences == [
        "First relevant statement.",
        "Second relevant statement.",
    ]