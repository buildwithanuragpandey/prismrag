import json

from adaptive_agentic_rag.orchestration.corpus_source_availability import (
    CorpusSourceAvailability,
)


def make_catalog(
    tmp_path,
):

    path = (
        tmp_path
        /
        "corpus.json"
    )


    payload = [
        {
            "id":
                "chunk_1",

            "metadata": {
                "source":
                    "The Age",
            },
        },

        {
            "id":
                "chunk_2",

            "metadata": {
                "source":
                    "The New York Times",
            },
        },

        {
            "id":
                "chunk_3",

            "metadata": {
                "source":
                    (
                        "Cnbc | "
                        "World Business News Leader"
                    ),
            },
        },

        {
            "id":
                "chunk_4",

            "metadata": {
                "source":
                    "CBSSports.com",
            },
        },
    ]


    path.write_text(
        json.dumps(
            payload
        ),
        encoding="utf-8",
    )


    return path


def test_exact_source_is_available(
    tmp_path,
):

    availability = (
        CorpusSourceAvailability(
            corpus_path=
                make_catalog(
                    tmp_path
                )
        )
    )


    result = (
        availability.check(
            [
                "The Age",
            ]
        )
    )


    assert (
        result.all_available
        is True
    )


    assert (
        result.unavailable_sources
        ==
        []
    )


def test_safe_alias_matches_catalog_source(
    tmp_path,
):

    availability = (
        CorpusSourceAvailability(
            corpus_path=
                make_catalog(
                    tmp_path
                )
        )
    )


    result = (
        availability.check(
            [
                "New York Times",
            ]
        )
    )


    assert (
        result.all_available
        is True
    )


    assert (
        result.matched_catalog_sources[
            "New York Times"
        ]
        ==
        [
            "The New York Times",
        ]
    )


def test_pipe_primary_alias_matches(
    tmp_path,
):

    availability = (
        CorpusSourceAvailability(
            corpus_path=
                make_catalog(
                    tmp_path
                )
        )
    )


    result = (
        availability.check(
            [
                "CNBC",
            ]
        )
    )


    assert (
        result.all_available
        is True
    )


def test_missing_source_is_unavailable(
    tmp_path,
):

    availability = (
        CorpusSourceAvailability(
            corpus_path=
                make_catalog(
                    tmp_path
                )
        )
    )


    result = (
        availability.check(
            [
                "Forbes",
            ]
        )
    )


    assert (
        result.all_available
        is False
    )


    assert (
        result.unavailable_sources
        ==
        [
            "Forbes",
        ]
    )


def test_mixed_availability_fails_all_available(
    tmp_path,
):

    availability = (
        CorpusSourceAvailability(
            corpus_path=
                make_catalog(
                    tmp_path
                )
        )
    )


    result = (
        availability.check(
            [
                "The Age",
                "Forbes",
            ]
        )
    )


    assert (
        result.all_available
        is False
    )


    assert (
        result.available_sources
        ==
        [
            "The Age",
        ]
    )


    assert (
        result.unavailable_sources
        ==
        [
            "Forbes",
        ]
    )