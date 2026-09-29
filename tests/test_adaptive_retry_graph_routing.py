from adaptive_agentic_rag.orchestration.graph import (
    route_after_evidence,
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


def make_state(
    *,
    evidence_sufficient: bool,
    evidence_reasons: list[str],
    retry_count: int = 0,
    max_retries: int = 1,
):

    return {
        "evidence_sufficient":
            evidence_sufficient,

        "evidence_reasons":
            evidence_reasons,

        "retry_count":
            retry_count,

        "max_retries":
            max_retries,
    }


def test_sufficient_evidence_routes_to_generate():

    state = make_state(
        evidence_sufficient=True,

        evidence_reasons=[
            "evidence_path=v2",
        ],
    )


    assert (
        route_after_evidence(
            state
        )
        ==
        "generate"
    )


def test_partial_source_miss_available_in_corpus_routes_to_rewrite():

    state = make_state(
        evidence_sufficient=False,

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
        route_after_evidence(
            state,

            source_availability=
                FakeSourceAvailability(
                    {
                        "The Age",
                    }
                ),
        )
        ==
        "rewrite"
    )


def test_partial_source_miss_unavailable_in_corpus_routes_to_abstain():

    state = make_state(
        evidence_sufficient=False,

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
        route_after_evidence(
            state,

            source_availability=
                FakeSourceAvailability(
                    {
                        "TechCrunch",
                    }
                ),
        )
        ==
        "abstain"
    )


def test_semantic_rescue_reject_routes_to_abstain():

    state = make_state(
        evidence_sufficient=False,

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
        route_after_evidence(
            state
        )
        ==
        "abstain"
    )


def test_zero_explicit_source_coverage_routes_to_abstain():

    state = make_state(
        evidence_sufficient=False,

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
        route_after_evidence(
            state
        )
        ==
        "abstain"
    )


def test_partial_source_miss_with_exhausted_budget_abstains():

    state = make_state(
        evidence_sufficient=False,

        retry_count=1,
        max_retries=1,

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
        route_after_evidence(
            state,

            source_availability=
                FakeSourceAvailability(
                    {
                        "The Age",
                    }
                ),
        )
        ==
        "abstain"
    )


def test_runtime_retry_budget_is_honored():

    state = make_state(
        evidence_sufficient=False,

        retry_count=1,
        max_retries=2,

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
        route_after_evidence(
            state,

            source_availability=
                FakeSourceAvailability(
                    {
                        "The Age",
                    }
                ),
        )
        ==
        "rewrite"
    )


def test_unknown_evidence_rejection_routes_to_abstain():

    state = make_state(
        evidence_sufficient=False,

        evidence_reasons=[
            "unknown_future_gate=true",
        ],
    )


    assert (
        route_after_evidence(
            state
        )
        ==
        "abstain"
    )