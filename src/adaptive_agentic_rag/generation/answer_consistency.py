import re

from dataclasses import dataclass


YES_NO_STARTERS = {
    "am",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "should",
    "was",
    "were",
    "will",
    "would",
}


MULTI_EVIDENCE_PATTERNS = (
    r"\bwhile\b",
    r"\bwhereas\b",
    r"\bcompared\s+to\b",
    r"\bcompared\s+with\b",
    r"\bin\s+contrast\s+to\b",
    r"\bbetween\b",
    r"\bboth\s+articles\b",
    r"\bboth\s+reports\b",
    r"\beach\s+article\b",
    r"\beach\s+report\b",
    r"\brespectively\b",
    r"\bconsistent\s+with\b.*\band\s+with\b",
)


ENTITY_TARGET_PATTERNS = (
    r"^\s*who\b",
    r"^\s*which\s+individual\b",
    r"^\s*which\s+person\b",
    r"^\s*which\s+company\b",
    r"^\s*which\s+organization\b",
    r"^\s*which\s+country\b",
    r"^\s*which\s+city\b",
    r"^\s*which\s+university\b",
    r"^\s*which\s+team\b",
    r"^\s*which\s+player\b",
    r"^\s*which\s+news\s+source\b",
    r"^\s*which\s+source\b",
    r"^\s*what\s+company\b",
    r"^\s*what\s+organization\b",
)


SOURCE_ANSWER_TARGET_PATTERNS = (
    r"\bwhich\s+news\s+source\b",
    r"\bwhich\s+source\b",
    r"\bwhat\s+news\s+source\b",
    r"\bwhat\s+source\b",
    r"\bwhich\s+publication\b",
)


@dataclass
class AnswerConsistencyResult:

    valid: bool

    answer_type: str

    reasons: list[str]

    unique_citation_count: int


class AnswerConsistencyGuard:
    """
    Experimental post-grounding consistency guard.

    IMPORTANT:

    This component is retained only as an ablation artifact.

    It is currently NOT part of the production generator
    pipeline.

    The class remains available so historical diagnostics,
    tests, and ablation results stay reproducible.
    """

    # ========================================================
    # Normalization
    # ========================================================

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        tokens = re.findall(
            r"[a-z0-9]+",
            (text or "").lower(),
        )


        return " ".join(
            tokens
        )


    # ========================================================
    # Question type
    # ========================================================

    @classmethod
    def _is_yes_no_question(
        cls,
        query: str,
    ) -> bool:

        tokens = re.findall(
            r"[a-z]+",
            (query or "").lower(),
        )


        if not tokens:

            return False


        return (
            tokens[
                0
            ]
            in
            YES_NO_STARTERS
        )


    @staticmethod
    def _is_entity_question(
        query: str,
    ) -> bool:

        return any(
            re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            for pattern
            in ENTITY_TARGET_PATTERNS
        )


    @staticmethod
    def _allows_source_answer(
        query: str,
    ) -> bool:

        return any(
            re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            for pattern
            in SOURCE_ANSWER_TARGET_PATTERNS
        )


    @staticmethod
    def _requires_multiple_evidence_items(
        query: str,
    ) -> bool:

        return any(
            re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            for pattern
            in MULTI_EVIDENCE_PATTERNS
        )


    # ========================================================
    # Yes / No direct-answer helper
    # ========================================================

    @classmethod
    def _yes_no_value(
        cls,
        answer: str,
    ) -> str | None:

        normalized = (
            cls._normalize(
                answer
            )
        )


        if (
            normalized
            ==
            "yes"
            or
            normalized.startswith(
                "yes "
            )
        ):

            return "yes"


        if (
            normalized
            ==
            "no"
            or
            normalized.startswith(
                "no "
            )
        ):

            return "no"


        return None


    # ========================================================
    # Entity support
    # ========================================================

    @classmethod
    def _answer_appears_in_claims(
        cls,
        direct_answer: str,
        relevant_claims,
    ) -> bool:

        answer = (
            cls._normalize(
                direct_answer
            )
        )


        if not answer:

            return False


        for claim in (
            relevant_claims
        ):

            claim_text = (
                cls._normalize(
                    getattr(
                        claim,
                        "claim",
                        "",
                    )
                )
            )


            if (
                answer
                in
                claim_text
            ):

                return True


        return False


    # ========================================================
    # Provenance/source trap
    # ========================================================

    @classmethod
    def _looks_like_provenance_answer(
        cls,
        query: str,
        direct_answer: str,
    ) -> bool:

        query_normalized = (
            cls._normalize(
                query
            )
        )


        answer_normalized = (
            cls._normalize(
                direct_answer
            )
        )


        if not answer_normalized:

            return False


        escaped = re.escape(
            answer_normalized
        )


        provenance_patterns = (
            rf"\barticles?\s+from\s+{escaped}\b",
            rf"\breports?\s+from\s+{escaped}\b",
            rf"\breported\s+by\s+{escaped}\b",
            rf"\bcovered\s+by\s+{escaped}\b",
            rf"\bmentioned\s+by\s+{escaped}\b",
            rf"\baccording\s+to\s+{escaped}\b",
            rf"\barticles?\s+by\s+{escaped}\b",
            rf"\breports?\s+by\s+{escaped}\b",
        )


        return any(
            re.search(
                pattern,
                query_normalized,
            )

            for pattern
            in provenance_patterns
        )


    # ========================================================
    # Main validation
    # ========================================================

    def validate(
        self,
        *,
        query: str,
        direct_answer: str,
        relevant_claims,
    ) -> AnswerConsistencyResult:

        reasons = []


        citation_ids = {
            getattr(
                claim,
                "citation_id",
                None,
            )

            for claim
            in relevant_claims

            if (
                getattr(
                    claim,
                    "citation_id",
                    None,
                )
                is not None
            )
        }


        unique_citation_count = (
            len(
                citation_ids
            )
        )


        # ====================================================
        # Yes / No
        # ====================================================

        if self._is_yes_no_question(
            query
        ):

            answer_type = (
                "yes_no"
            )


            yes_no = (
                self._yes_no_value(
                    direct_answer
                )
            )


            if yes_no is None:

                reasons.append(
                    (
                        "Yes/no question did not produce "
                        "a Yes/No direct answer."
                    )
                )


            if (
                self._requires_multiple_evidence_items(
                    query
                )
                and
                unique_citation_count
                <
                2
            ):

                reasons.append(
                    (
                        "Comparison-style question lacks "
                        "multiple independently grounded "
                        "evidence citations."
                    )
                )


            return AnswerConsistencyResult(
                valid=(
                    len(
                        reasons
                    )
                    ==
                    0
                ),

                answer_type=
                    answer_type,

                reasons=
                    reasons,

                unique_citation_count=
                    unique_citation_count,
            )


        # ====================================================
        # Entity
        # ====================================================

        if self._is_entity_question(
            query
        ):

            answer_type = (
                "entity"
            )


            if not self._answer_appears_in_claims(
                direct_answer=
                    direct_answer,

                relevant_claims=
                    relevant_claims,
            ):

                reasons.append(
                    (
                        "Direct entity answer does not appear "
                        "in any verified relevant claim."
                    )
                )


            if (
                not self._allows_source_answer(
                    query
                )
                and
                self._looks_like_provenance_answer(
                    query=
                        query,

                    direct_answer=
                        direct_answer,
                )
            ):

                reasons.append(
                    (
                        "Direct entity answer appears to be "
                        "a publisher/source named only as "
                        "evidence provenance."
                    )
                )


            return AnswerConsistencyResult(
                valid=(
                    len(
                        reasons
                    )
                    ==
                    0
                ),

                answer_type=
                    answer_type,

                reasons=
                    reasons,

                unique_citation_count=
                    unique_citation_count,
            )


        # ====================================================
        # Other answer types
        # ====================================================

        return AnswerConsistencyResult(
            valid=
                True,

            answer_type=
                "other",

            reasons=
                [],

            unique_citation_count=
                unique_citation_count,
        )