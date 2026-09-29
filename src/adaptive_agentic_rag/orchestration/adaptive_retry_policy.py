import ast

from dataclasses import dataclass
from enum import Enum

from adaptive_agentic_rag.orchestration.corpus_source_availability import (
    CorpusSourceAvailability,
)


class RetryAction(
    str,
    Enum,
):

    GENERATE = "generate"

    RETRY = "retry"

    ABSTAIN = "abstain"


@dataclass
class RetryDecision:

    action: RetryAction

    reason: str

    evidence_path: str | None

    required_sources: list[str]

    covered_sources: list[str]

    missing_sources: list[str]

    retry_count: int


class AdaptiveRetryPolicy:
    """
    Conservative retry policy.

    Goal
    ----
    Retry retrieval only when another retrieval attempt is
    structurally possible and justified.

    Safe rules
    ----------

    1. Evidence sufficient
       -> GENERATE

    2. Retry budget exhausted
       -> ABSTAIN

    3. Explicit source coverage rejection with PARTIAL
       source coverage:

       3a. Missing source unavailable in corpus
           -> ABSTAIN

       3b. Missing source available in corpus
           -> RETRY

    4. Explicit source coverage rejection with ZERO
       requested sources covered
       -> ABSTAIN

    5. Source coverage satisfied but constrained semantic
       rescue rejects
       -> ABSTAIN

    6. Unknown rejection state
       -> ABSTAIN

    Important
    ---------
    The policy does NOT use:

    - gold evidence
    - dataset labels
    - null-query labels
    - evidence-score thresholds
    - embedding thresholds
    - LLM classification

    Source availability is corpus-structural metadata only.
    """

    def __init__(
        self,
        *,
        max_retries: int = 1,
        source_availability: CorpusSourceAvailability | None = None,
    ):

        self.max_retries = (
            max_retries
        )


        # ----------------------------------------------------
        # Dependency is optional so the policy remains easy
        # to unit-test and backwards-compatible when used
        # outside the production graph.
        #
        # The production graph always injects it.
        # ----------------------------------------------------

        self.source_availability = (
            source_availability
        )


    # ========================================================
    # Telemetry helpers
    # ========================================================

    @staticmethod
    def _extract_value(
        reasons: list[str],
        prefix: str,
    ) -> str | None:

        for reason in (
            reasons
            or []
        ):

            if not isinstance(
                reason,
                str,
            ):

                continue


            if reason.startswith(
                prefix
            ):

                return (
                    reason[
                        len(
                            prefix
                        ):
                    ]
                    .strip()
                )


        return None


    @classmethod
    def _extract_list(
        cls,
        reasons: list[str],
        prefix: str,
    ) -> list[str]:

        raw_value = (
            cls._extract_value(
                reasons=
                    reasons,

                prefix=
                    prefix,
            )
        )


        if raw_value is None:

            return []


        try:

            parsed = (
                ast.literal_eval(
                    raw_value
                )
            )


        except (
            ValueError,
            SyntaxError,
        ):

            return []


        if not isinstance(
            parsed,
            list,
        ):

            return []


        return [
            str(
                item
            )

            for item
            in parsed
        ]


    @classmethod
    def _extract_path(
        cls,
        reasons: list[str],
    ) -> str | None:

        return (
            cls._extract_value(
                reasons=
                    reasons,

                prefix=
                    "evidence_path=",
            )
        )


    # ========================================================
    # Decision factory
    # ========================================================

    @staticmethod
    def _decision(
        *,
        action: RetryAction,
        reason: str,
        evidence_path: str | None,
        required_sources: list[str],
        covered_sources: list[str],
        missing_sources: list[str],
        retry_count: int,
    ) -> RetryDecision:

        return RetryDecision(
            action=
                action,

            reason=
                reason,

            evidence_path=
                evidence_path,

            required_sources=
                required_sources,

            covered_sources=
                covered_sources,

            missing_sources=
                missing_sources,

            retry_count=
                retry_count,
        )


    # ========================================================
    # Public policy
    # ========================================================

    def decide(
        self,
        *,
        evidence_sufficient: bool,
        retry_count: int,
        evidence_reasons: list[str],
    ) -> RetryDecision:

        evidence_path = (
            self._extract_path(
                evidence_reasons
            )
        )


        required_sources = (
            self._extract_list(
                evidence_reasons,
                "required_sources=",
            )
        )


        covered_sources = (
            self._extract_list(
                evidence_reasons,
                "covered_sources=",
            )
        )


        missing_sources = (
            self._extract_list(
                evidence_reasons,
                "missing_sources=",
            )
        )


        # ====================================================
        # Evidence already accepted
        # ====================================================

        if evidence_sufficient:

            return self._decision(
                action=
                    RetryAction.GENERATE,

                reason=(
                    "Evidence is already sufficient."
                ),

                evidence_path=
                    evidence_path,

                required_sources=
                    required_sources,

                covered_sources=
                    covered_sources,

                missing_sources=
                    missing_sources,

                retry_count=
                    retry_count,
            )


        # ====================================================
        # Retry budget exhausted
        # ====================================================

        if (
            retry_count
            >=
            self.max_retries
        ):

            return self._decision(
                action=
                    RetryAction.ABSTAIN,

                reason=(
                    "Retry budget exhausted."
                ),

                evidence_path=
                    evidence_path,

                required_sources=
                    required_sources,

                covered_sources=
                    covered_sources,

                missing_sources=
                    missing_sources,

                retry_count=
                    retry_count,
            )


        # ====================================================
        # Explicit source coverage failure
        # ====================================================

        if (
            evidence_path
            ==
            "explicit_source_coverage_reject"
        ):

            # ------------------------------------------------
            # Partial source coverage is the only currently
            # approved retrieval-retry family.
            # ------------------------------------------------

            if (
                missing_sources
                and
                covered_sources
            ):

                # ============================================
                # Corpus availability invariant
                #
                # If production supplied the availability
                # dependency, a missing publisher that does
                # not exist anywhere in the corpus cannot be
                # recovered by query rewriting.
                # ============================================

                if (
                    self.source_availability
                    is not None
                ):

                    availability = (
                        self.source_availability.check(
                            missing_sources
                        )
                    )


                    if not (
                        availability.all_available
                    ):

                        unavailable = (
                            availability
                            .unavailable_sources
                        )


                        return self._decision(
                            action=
                                RetryAction.ABSTAIN,

                            reason=(
                                "Explicit source evidence is "
                                "missing, but retrieval cannot "
                                "recover all missing sources "
                                "because they are unavailable "
                                "in the corpus: "
                                f"{unavailable!r}."
                            ),

                            evidence_path=
                                evidence_path,

                            required_sources=
                                required_sources,

                            covered_sources=
                                covered_sources,

                            missing_sources=
                                missing_sources,

                            retry_count=
                                retry_count,
                        )


                # ============================================
                # Missing source exists in corpus.
                #
                # Another retrieval attempt is structurally
                # possible.
                # ============================================

                return self._decision(
                    action=
                        RetryAction.RETRY,

                    reason=(
                        "Partial explicit-source coverage "
                        "with corpus-available missing "
                        "sources suggests a recoverable "
                        "retrieval miss."
                    ),

                    evidence_path=
                        evidence_path,

                    required_sources=
                        required_sources,

                    covered_sources=
                        covered_sources,

                    missing_sources=
                        missing_sources,

                    retry_count=
                        retry_count,
                )


            # ------------------------------------------------
            # Zero explicit-source coverage.
            # ------------------------------------------------

            return self._decision(
                action=
                    RetryAction.ABSTAIN,

                reason=(
                    "Explicit-source coverage failed without "
                    "partial source support; blind retry is "
                    "not justified."
                ),

                evidence_path=
                    evidence_path,

                required_sources=
                    required_sources,

                covered_sources=
                    covered_sources,

                missing_sources=
                    missing_sources,

                retry_count=
                    retry_count,
            )


        # ====================================================
        # Semantic rescue failed after source coverage.
        # ====================================================

        if (
            evidence_path
            ==
            "constrained_semantic_rescue_reject"
        ):

            return self._decision(
                action=
                    RetryAction.ABSTAIN,

                reason=(
                    "Evidence remained insufficient after "
                    "source coverage and constrained semantic "
                    "rescue; no structural retrieval-miss "
                    "signal is present."
                ),

                evidence_path=
                    evidence_path,

                required_sources=
                    required_sources,

                covered_sources=
                    covered_sources,

                missing_sources=
                    missing_sources,

                retry_count=
                    retry_count,
            )


        # ====================================================
        # Conservative fallback
        # ====================================================

        return self._decision(
            action=
                RetryAction.ABSTAIN,

            reason=(
                "No production-visible signal justifies "
                "another retrieval attempt."
            ),

            evidence_path=
                evidence_path,

            required_sources=
                required_sources,

            covered_sources=
                covered_sources,

            missing_sources=
                missing_sources,

            retry_count=
                retry_count,
        )