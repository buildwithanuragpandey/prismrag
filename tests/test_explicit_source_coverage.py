from types import SimpleNamespace

from adaptive_agentic_rag.orchestration.explicit_source_coverage import (
    ExplicitSourceCoverageGuard,
)


# ============================================================
# Test context helper
# ============================================================

def context(
    *sources,
):

    return SimpleNamespace(
        items=[
            SimpleNamespace(
                source=source
            )

            for source
            in sources
        ]
    )


# ============================================================
# Existing safety invariants
# ============================================================

def test_case15_missing_the_age_fails():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Has the portrayal of Google's market practices "
        "in reports by The Age after October 22, 2023, "
        "remained consistent with the depiction in "
        "The Verge's coverage of the Epic v. Google case, "
        "and with TechCrunch's report on the class action "
        "antitrust suit filed against Google?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "The Verge",
                "TechCrunch",
            ),
    )


    assert (
        result.satisfied
        is False
    )


    assert any(
        (
            guard._normalize(
                source
            )
            ==
            "the age"
        )

        for source
        in result.missing_sources
    )


    assert (
        "The Verge"
        in
        result.covered_sources
    )


    assert (
        "TechCrunch"
        in
        result.covered_sources
    )


def test_case15_all_sources_present_passes():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Has the portrayal of Google's market practices "
        "in reports by The Age after October 22, 2023, "
        "remained consistent with the depiction in "
        "The Verge's coverage of the Epic v. Google case, "
        "and with TechCrunch's report on the class action "
        "antitrust suit filed against Google?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "The Age",
                "The Verge",
                "TechCrunch",
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


def test_case10_three_sources_present_passes():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Does the Mashable article suggest that Amazon's "
        "Cyber Monday includes both continued and new deals, "
        "while The Sydney Morning Herald article focuses on "
        "the impact of an antitrust lawsuit on Amazon's "
        "stock price, and the Cnbc | World Business News "
        "Leader article discusses the opportunity of "
        "selling on Amazon?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "Mashable",
                "The Sydney Morning Herald",
                (
                    "Cnbc | "
                    "World Business News Leader"
                ),
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


def test_pipe_source_can_match_primary_alias():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "According to CNBC, Amazon sellers "
        "discussed their experience."
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                (
                    "Cnbc | "
                    "World Business News Leader"
                )
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        (
            "Cnbc | "
            "World Business News Leader"
        )
        in
        result.covered_sources
    )


def test_quoted_pipe_source_is_supported():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Between the report from "
        "'The Roar | Sports Writers Blog' "
        "on October 19, 2023 and another update, "
        "which source reported the larger lead?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                (
                    "The Roar | "
                    "Sports Writers Blog"
                )
            ),
    )


    assert (
        result.satisfied
        is True
    )


def test_missing_single_explicit_source_fails():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "According to Reuters, "
        "which country launched the initiative?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "Bloomberg"
            ),
    )


    assert (
        result.satisfied
        is False
    )


    assert any(
        (
            guard._normalize(
                source
            )
            ==
            "reuters"
        )

        for source
        in result.missing_sources
    )


def test_query_without_explicit_source_passes():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Which company faced an "
        "antitrust lawsuit?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "TechCrunch",
                "The Verge",
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


def test_reports_by_source_followed_by_remained_is_detected():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Has Google's portrayal in reports by "
        "The Age remained consistent with "
        "The Verge's coverage and "
        "TechCrunch's report?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "The Verge",
                "TechCrunch",
            ),
    )


    assert (
        result.satisfied
        is False
    )


    assert any(
        (
            "the age"
            ==
            guard._normalize(
                source
            )
        )

        for source
        in result.missing_sources
    )


    assert (
        "The Verge"
        in
        result.covered_sources
    )


    assert (
        "TechCrunch"
        in
        result.covered_sources
    )


# ============================================================
# Frozen500 regression family:
# reference phrases are NOT publishers
# ============================================================

def test_the_subsequent_is_not_a_source():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "After the report by The Age on October 22, 2023, "
        "and the subsequent report by TechCrunch on "
        "October 31, 2023, was the reporting consistent?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "The Age",
                "TechCrunch",
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


    assert not any(
        "subsequent"
        in source.lower()

        for source
        in result.required_sources
    )


def test_the_other_is_not_a_source():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Does the article from "
        "The Roar | Sports Writers Blog "
        "about the Sydney Kings differ from the other "
        "article from The Roar | Sports Writers Blog "
        "about Eddie Jones?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                (
                    "The Roar | "
                    "Sports Writers Blog"
                )
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


    assert not any(
        (
            guard._normalize(
                source
            )
            ==
            "the other"
        )

        for source
        in result.required_sources
    )


def test_both_sources_is_not_a_source():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Considering reports from The New York Times "
        "and The Guardian, which actor according to "
        "both sources played the lead guitarist?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "The New York Times",
                "The Guardian",
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


def test_these_sources_is_not_a_source():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "After the Fortune article and the TechCrunch "
        "article, was the portrayal consistent according "
        "to these sources?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "Fortune",
                "TechCrunch",
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


def test_same_source_is_not_a_new_source():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Between the TechCrunch report on the trial "
        "and the subsequent report by the same source, "
        "was the portrayal consistent?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "TechCrunch"
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


# ============================================================
# Frozen500 regression family:
# compound publisher references
# ============================================================

def test_articles_from_two_sources_resolve_to_two_publishers():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Who is the individual facing criminal charges "
        "according to articles from TechCrunch and "
        "The Verge?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "TechCrunch",
                "The Verge",
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


    assert (
        "TechCrunch"
        in
        result.required_sources
    )


    assert (
        "The Verge"
        in
        result.required_sources
    )


def test_compound_reference_can_detect_one_real_missing_source():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Who is the individual facing criminal charges "
        "according to articles from TechCrunch and "
        "The Verge?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "TechCrunch"
            ),
    )


    assert (
        result.satisfied
        is False
    )


    assert (
        "TechCrunch"
        in
        result.covered_sources
    )


    assert any(
        (
            guard._normalize(
                source
            )
            ==
            "the verge"
        )

        for source
        in result.missing_sources
    )


def test_in_contrast_to_source_is_cleaned():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Does the Sporting News article suggest one "
        "outcome, in contrast to the CBSSports.com "
        "article which discusses another outcome?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "Sporting News",
                "CBSSports.com",
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


# ============================================================
# Frozen500 regression family:
# publisher followed by descriptive text
# ============================================================

def test_reports_from_source_between_dates_does_not_capture_date():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Is the involvement of Sony Music artists "
        "consistent according to reports from "
        "Music Business Worldwide between November 23, "
        "2023 and November 30, 2023?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "Music Business Worldwide"
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


def test_source_followed_by_defending_clause_is_trimmed():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Does the article from The Verge defending "
        "Apple's Google Search deal suggest there was "
        "no valid alternative?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "The Verge"
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


# ============================================================
# Leading "The" alias regression
# ============================================================

def test_guardian_alias_matches_the_guardian():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Does the Guardian article discuss "
        "the team's position?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "The Guardian"
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


def test_the_age_does_not_get_unsafe_age_alias():

    aliases = (
        ExplicitSourceCoverageGuard
        ._source_aliases(
            "The Age"
        )
    )


    assert (
        "the age"
        in aliases
    )


    assert (
        "age"
        not in aliases
    )
def test_according_to_person_is_not_treated_as_source():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Which company, according to Eddy Cue, had no "
        "valid alternative for search engine services, "
        "is reported by The Verge to have spent billions "
        "to remain the default search engine, as mentioned "
        "by TechCrunch?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "The Verge",
                "TechCrunch",
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


    assert not any(
        (
            guard._normalize(
                source
            )
            ==
            "eddy cue"
        )

        for source
        in result.required_sources
    )


def test_according_to_missing_publisher_still_fails():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "According to Reuters, "
        "which country launched the initiative?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "Bloomberg"
            ),
    )


    assert (
        result.satisfied
        is False
    )


    assert any(
        (
            guard._normalize(
                source
            )
            ==
            "reuters"
        )

        for source
        in result.missing_sources
    )


def test_quoted_compound_sources_are_split_correctly():

    guard = (
        ExplicitSourceCoverageGuard()
    )


    query = (
        "Has the focus of Taylor Swift coverage by "
        "'The Independent - Life and Style' and "
        "'FOX News - Lifestyle' remained consistent?"
    )


    result = guard.check(
        query=
            query,

        context=
            context(
                "The Independent - Life and Style",
                "FOX News - Lifestyle",
            ),
    )


    assert (
        result.satisfied
        is True
    )


    assert (
        result.missing_sources
        ==
        []
    )


    assert (
        "The Independent - Life and Style"
        in
        result.required_sources
    )


    assert (
        "FOX News - Lifestyle"
        in
        result.required_sources
    )


    assert not any(
        (
            " and "
            in source.lower()
        )

        for source
        in result.missing_sources
    )    