from adaptive_agentic_rag.orchestration.adaptive_retry_policy import (
    AdaptiveRetryPolicy,
    RetryAction,
)

from adaptive_agentic_rag.orchestration.corpus_source_availability import (
    CorpusSourceAvailabilityResult,
)


class FakeSourceAvailability:

    def __init__(
        self,
        available_sources,
    ):

        self.available_sources = set(
            available_sources
        )


    def check(
        self,
        sources,
    ):

        available = [
            source

            for source
            in sources

            if source
            in self.available_sources
        ]


        unavailable = [
            source

            for source
            in sources

            if source
            not in self.available_sources
        ]


        return CorpusSourceAvailabilityResult(
            requested_sources=
                list(
                    sources
                ),

            available_sources=
                available,

            unavailable_sources=
                unavailable,

            matched_catalog_sources={
                source: (
                    [
                        source
                    ]

                    if source
                    in self.available_sources

                    else []
                )

                for source
                in sources
            },
        )


def policy(
    *,
    available_sources=None,
):

    source_availability = None


    if (
        available_sources
        is not None
    ):

        source_availability = (
            FakeSourceAvailability(
                available_sources
            )
        )


    return AdaptiveRetryPolicy(
        max_retries=1,
        source_availability=
            source_availability,
    )


def test_sufficient_evidence_goes_to_generation():

    result = policy().decide(
        evidence_sufficient=True,
        retry_count=0,
        evidence_reasons=[
            "evidence_path=v2",
        ],
    )


    assert (
        result.action
        ==
        RetryAction.GENERATE
    )


def test_partial_explicit_source_miss_available_in_corpus_allows_retry():

    result = policy(
        available_sources={
            "The Age",
        },
    ).decide(
        evidence_sufficient=False,
        retry_count=0,
        evidence_reasons=[
            (
                "required_sources="
                "['The Age', 'The Verge', 'TechCrunch']"
            ),

            (
                "covered_sources="
                "['The Verge', 'TechCrunch']"
            ),

            (
                "missing_sources="
                "['The Age']"
            ),

            (
                "evidence_path="
                "explicit_source_coverage_reject"
            ),
        ],
    )


    assert (
        result.action
        ==
        RetryAction.RETRY
    )


def test_partial_explicit_source_miss_unavailable_in_corpus_abstains():

    result = policy(
        available_sources={
            "TechCrunch",
        },
    ).decide(
        evidence_sufficient=False,
        retry_count=0,
        evidence_reasons=[
            (
                "required_sources="
                "['TechCrunch', 'Forbes']"
            ),

            (
                "covered_sources="
                "['TechCrunch']"
            ),

            (
                "missing_sources="
                "['Forbes']"
            ),

            (
                "evidence_path="
                "explicit_source_coverage_reject"
            ),
        ],
    )


    assert (
        result.action
        ==
        RetryAction.ABSTAIN
    )


    assert (
        "unavailable"
        in
        result.reason.lower()
    )


def test_multiple_missing_sources_require_all_to_exist():

    result = policy(
        available_sources={
            "The Age",
        },
    ).decide(
        evidence_sufficient=False,
        retry_count=0,
        evidence_reasons=[
            (
                "required_sources="
                "['TechCrunch', 'The Age', 'Forbes']"
            ),

            (
                "covered_sources="
                "['TechCrunch']"
            ),

            (
                "missing_sources="
                "['The Age', 'Forbes']"
            ),

            (
                "evidence_path="
                "explicit_source_coverage_reject"
            ),
        ],
    )


    assert (
        result.action
        ==
        RetryAction.ABSTAIN
    )


def test_all_explicit_sources_missing_does_not_retry():

    result = policy().decide(
        evidence_sufficient=False,
        retry_count=0,
        evidence_reasons=[
            (
                "required_sources="
                "['Bloomberg', 'CNN']"
            ),

            "covered_sources=[]",

            (
                "missing_sources="
                "['Bloomberg', 'CNN']"
            ),

            (
                "evidence_path="
                "explicit_source_coverage_reject"
            ),
        ],
    )


    assert (
        result.action
        ==
        RetryAction.ABSTAIN
    )


def test_complete_source_coverage_semantic_reject_does_not_retry():

    result = policy().decide(
        evidence_sufficient=False,
        retry_count=0,
        evidence_reasons=[
            (
                "explicit_source_coverage="
                "satisfied"
            ),

            (
                "required_sources="
                "['The Verge', 'TechCrunch']"
            ),

            (
                "covered_sources="
                "['The Verge', 'TechCrunch']"
            ),

            "semantic_rescue=insufficient",

            (
                "evidence_path="
                "constrained_semantic_rescue_reject"
            ),
        ],
    )


    assert (
        result.action
        ==
        RetryAction.ABSTAIN
    )


def test_retry_budget_prevents_second_retry():

    result = policy(
        available_sources={
            "The Age",
        },
    ).decide(
        evidence_sufficient=False,
        retry_count=1,
        evidence_reasons=[
            (
                "required_sources="
                "['The Age', 'TechCrunch']"
            ),

            (
                "covered_sources="
                "['TechCrunch']"
            ),

            (
                "missing_sources="
                "['The Age']"
            ),

            (
                "evidence_path="
                "explicit_source_coverage_reject"
            ),
        ],
    )


    assert (
        result.action
        ==
        RetryAction.ABSTAIN
    )


def test_unknown_rejection_is_fail_closed():

    result = policy().decide(
        evidence_sufficient=False,
        retry_count=0,
        evidence_reasons=[
            "some_future_reason=true",
        ],
    )


    assert (
        result.action
        ==
        RetryAction.ABSTAIN
    )


def test_malformed_telemetry_does_not_crash():

    result = policy().decide(
        evidence_sufficient=False,
        retry_count=0,
        evidence_reasons=[
            "required_sources=not-a-list",
            "covered_sources=[broken",
            "missing_sources=???",
            (
                "evidence_path="
                "explicit_source_coverage_reject"
            ),
        ],
    )


    assert (
        result.action
        ==
        RetryAction.ABSTAIN
    )