from adaptive_agentic_rag.orchestration.constrained_semantic_rescue import (
    ConstrainedSemanticRescue,
)


def make_rescue():

    # These tests exercise pure structural helpers.
    # No model needs to be loaded.
    return ConstrainedSemanticRescue(
        reranker=None,
        evidence_grader=None,
    )


def test_query_source_completeness_detects_missing_wsj_and_bloomberg():

    rescue = make_rescue()


    candidates = [
        {
            "source":
                "The Sydney Morning Herald"
        }
    ]


    query = (
        "Considering the recent fluctuations "
        "in the Dow Jones Industrial Average "
        "as reported by The Wall Street Journal "
        "and the impact of tech stocks on the index "
        "as detailed by Bloomberg, which company "
        "had significant influence?"
    )


    (
        required,
        missing,
    ) = rescue._missing_query_sources(
        query=query,
        candidates=candidates,
    )


    required_signatures = {
        rescue._source_signature(
            value
        )
        for value
        in required
    }


    missing_signatures = {
        rescue._source_signature(
            value
        )
        for value
        in missing
    }


    assert (
        "wallstreetjournal"
        in
        required_signatures
    )


    assert (
        "bloomberg"
        in
        required_signatures
    )


    assert (
        "wallstreetjournal"
        in
        missing_signatures
    )


    assert (
        "bloomberg"
        in
        missing_signatures
    )


def test_person_bridge_accepts_same_person_across_requirements():

    rescue = make_rescue()


    result = rescue._person_bridge(

        query=(
            "Who was associated with ChatGPT "
            "and later discussed by a company board?"
        ),

        supported_requirements=[
            {
                "text":
                    "associated with ChatGPT",

                "best_document_id":
                    "doc_a",

                "best_title":
                    "Sam Altman and ChatGPT",

                "best_raw_text":
                    (
                        "Sam Altman became closely "
                        "associated with ChatGPT."
                    ),
            },
            {
                "text":
                    "discussed by the board",

                "best_document_id":
                    "doc_b",

                "best_title":
                    "OpenAI board discusses Sam Altman",

                "best_raw_text":
                    (
                        "The board said Sam Altman "
                        "had not been consistently candid."
                    ),
            },
        ],

        required_count=2,
    )


    assert (
        result[
            "ok"
        ]
        is True
    )


    assert (
        "sam altman"
        in
        result[
            "candidate_people"
        ]
    )


def test_person_bridge_rejects_different_people():

    rescue = make_rescue()


    result = rescue._person_bridge(

        query=(
            "Who was associated with ChatGPT "
            "and later discussed by a company board?"
        ),

        supported_requirements=[
            {
                "text":
                    "associated with ChatGPT",

                "best_document_id":
                    "doc_a",

                "best_title":
                    "Elon Musk discusses AI",

                "best_raw_text":
                    "Elon Musk discussed artificial intelligence.",
            },
            {
                "text":
                    "discussed by the board",

                "best_document_id":
                    "doc_b",

                "best_title":
                    "OpenAI board discusses Sam Altman",

                "best_raw_text":
                    (
                        "The board discussed "
                        "Sam Altman."
                    ),
            },
        ],

        required_count=2,
    )


    assert (
        result[
            "ok"
        ]
        is False
    )


    assert (
        result[
            "candidate_people"
        ]
        ==
        []
    )