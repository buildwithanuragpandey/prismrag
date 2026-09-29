import re

from dataclasses import dataclass


STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "for",
    "from",
    "to",
    "in",
    "on",
    "at",
    "by",
    "with",
    "and",
    "or",
    "also",
    "was",
    "were",
    "is",
    "are",
    "has",
    "have",
    "had",
    "did",
    "does",
    "do",
    "that",
    "this",
    "these",
    "those",
    "their",
    "its",
}


NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "without",
    "neither",
    "nor",
    "cannot",
    "cant",
    "didnt",
    "doesnt",
    "isnt",
    "wasnt",
    "werent",
    "hasnt",
    "havent",
}


@dataclass
class RelationResolution:

    applied: bool

    resolved_answer: str | None

    relation_type: str | None

    requested_polarity: str | None

    fact_count: int

    predicate_signatures: list[
        tuple[str, ...]
    ]

    reason: str


class RelationAwareAnswerResolver:
    """
    Conservative deterministic answer resolver.

    Current supported relation:
        consistency

    Resolution is intentionally low-coverage.

    It only resolves when two or more grounded facts reduce
    to EXACTLY the same predicate signature after removing
    quoted entity/event spans.

    Safe rules:

        equivalent predicates
        + positive consistency question
        -> Yes

        equivalent predicates
        + explicit inconsistency question
        -> No

    Anything else:
        no override

    We deliberately do NOT infer inconsistency merely because
    two facts differ.
    """

    def __init__(
        self,
        *,
        minimum_signature_tokens: int = 4,
    ):

        self.minimum_signature_tokens = (
            minimum_signature_tokens
        )


    # ========================================================
    # Relation detection
    # ========================================================

    @staticmethod
    def _detect_relation(
        query: str,
    ) -> tuple[
        str | None,
        str | None,
    ]:

        normalized = (
            query
            or ""
        ).lower()


        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip()


        # ====================================================
        # IMPORTANT:
        #
        # Negative/inconsistency patterns MUST be evaluated
        # before positive consistency patterns.
        #
        # Because:
        #
        #     "inconsistent with"
        #
        # contains:
        #
        #     "consistent with"
        #
        # Checking positive first would misclassify:
        #
        #     Were the reports inconsistent with each other?
        #
        # as a positive consistency query.
        # ====================================================

        negative_patterns = [
            r"\binconsistent\s+with\b",
            r"\bwas\s+inconsistent\b",
            r"\bwere\s+inconsistent\b",
            r"\bis\s+inconsistent\b",
            r"\bare\s+inconsistent\b",
            r"\bbecome\s+inconsistent\b",
            r"\bbecame\s+inconsistent\b",
        ]


        for pattern in negative_patterns:

            if re.search(
                pattern,
                normalized,
            ):

                return (
                    "consistency",
                    "negative",
                )


        # ====================================================
        # Positive consistency
        # ====================================================

        positive_patterns = [
            r"\bremained\s+consistent\b",
            r"\bremain\s+consistent\b",
            r"\bremains\s+consistent\b",
            r"\bconsistent\s+perspective\b",
            r"\bconsistent\s+with\b",
            r"\bshow\s+a\s+consistent\b",
            r"\bshows\s+a\s+consistent\b",
            r"\bshowed\s+a\s+consistent\b",
            r"\bcommon\s+trend\b",
        ]


        for pattern in positive_patterns:

            if re.search(
                pattern,
                normalized,
            ):

                return (
                    "consistency",
                    "positive",
                )


        return (
            None,
            None,
        )


    # ========================================================
    # Remove quoted entities/events
    #
    # Example:
    #
    # "Jaguars vs. Saints"
    # "Chiefs vs. Packers"
    #
    # These are argument values rather than the predicate
    # being compared.
    # ========================================================

    @staticmethod
    def _remove_quoted_spans(
        text: str,
    ) -> str:

        text = (
            text
            or ""
        )


        text = re.sub(
            r'"[^"]*"',
            " ",
            text,
        )


        text = re.sub(
            r"'[^']*'",
            " ",
            text,
        )


        text = re.sub(
            r"“[^”]*”",
            " ",
            text,
        )


        text = re.sub(
            r"‘[^’]*’",
            " ",
            text,
        )


        return text


    # ========================================================
    # Apostrophe normalization
    # ========================================================

    @staticmethod
    def _normalize_apostrophes(
        text: str,
    ) -> str:

        return (
            text
            .replace(
                "’",
                "'",
            )
            .replace(
                "‘",
                "'",
            )
        )


    # ========================================================
    # Predicate signature
    # ========================================================

    def _predicate_signature(
        self,
        fact: str,
    ) -> tuple[str, ...]:

        text = (
            self._normalize_apostrophes(
                fact
                or ""
            )
        )


        text = (
            self._remove_quoted_spans(
                text
            )
        )


        text = (
            text.lower()
        )


        # ----------------------------------------------------
        # Possessive normalization
        #
        # player's -> player
        # ----------------------------------------------------

        text = re.sub(
            r"\b([a-z0-9]+)'s\b",
            r"\1",
            text,
        )


        # ----------------------------------------------------
        # Contraction normalization while keeping negation.
        #
        # didn't -> didnt
        # isn't  -> isnt
        # ----------------------------------------------------

        text = re.sub(
            r"n't\b",
            "nt",
            text,
        )


        tokens = re.findall(
            r"[a-z0-9]+",
            text,
        )


        signature = []


        for token in tokens:

            if token in STOPWORDS:

                continue


            if token.isdigit():

                continue


            signature.append(
                token
            )


        return tuple(
            signature
        )


    # ========================================================
    # Negation signature
    # ========================================================

    @staticmethod
    def _negation_signature(
        signature: tuple[str, ...],
    ) -> tuple[str, ...]:

        return tuple(
            token

            for token
            in signature

            if token
            in NEGATION_TERMS
        )


    # ========================================================
    # Exact predicate equivalence
    # ========================================================

    def _all_predicates_equivalent(
        self,
        signatures: list[
            tuple[str, ...]
        ],
    ) -> bool:

        if len(
            signatures
        ) < 2:

            return False


        for signature in signatures:

            if (
                len(
                    signature
                )
                <
                self.minimum_signature_tokens
            ):

                return False


        reference = (
            signatures[
                0
            ]
        )


        reference_negation = (
            self._negation_signature(
                reference
            )
        )


        for signature in (
            signatures[
                1:
            ]
        ):

            # ------------------------------------------------
            # Exact match only.
            #
            # No lexical threshold.
            # No embedding threshold.
            # ------------------------------------------------

            if signature != reference:

                return False


            # ------------------------------------------------
            # Explicit negation invariant.
            # ------------------------------------------------

            if (
                self._negation_signature(
                    signature
                )
                !=
                reference_negation
            ):

                return False


        return True


    # ========================================================
    # Public resolver
    # ========================================================

    def resolve(
        self,
        *,
        query: str,
        facts: list[str],
    ) -> RelationResolution:

        relation_type, polarity = (
            self._detect_relation(
                query
            )
        )


        signatures = [
            self._predicate_signature(
                fact
            )

            for fact
            in facts
        ]


        # ====================================================
        # Unsupported relation
        # ====================================================

        if relation_type is None:

            return RelationResolution(
                applied=
                    False,

                resolved_answer=
                    None,

                relation_type=
                    None,

                requested_polarity=
                    None,

                fact_count=
                    len(
                        facts
                    ),

                predicate_signatures=
                    signatures,

                reason=(
                    "No supported deterministic "
                    "relation type detected."
                ),
            )


        # ====================================================
        # Need at least two facts
        # ====================================================

        if len(
            facts
        ) < 2:

            return RelationResolution(
                applied=
                    False,

                resolved_answer=
                    None,

                relation_type=
                    relation_type,

                requested_polarity=
                    polarity,

                fact_count=
                    len(
                        facts
                    ),

                predicate_signatures=
                    signatures,

                reason=(
                    "At least two grounded facts "
                    "are required."
                ),
            )


        # ====================================================
        # Exact predicate equivalence
        # ====================================================

        equivalent = (
            self._all_predicates_equivalent(
                signatures
            )
        )


        if not equivalent:

            return RelationResolution(
                applied=
                    False,

                resolved_answer=
                    None,

                relation_type=
                    relation_type,

                requested_polarity=
                    polarity,

                fact_count=
                    len(
                        facts
                    ),

                predicate_signatures=
                    signatures,

                reason=(
                    "Grounded facts do not have "
                    "exact equivalent predicate "
                    "signatures."
                ),
            )


        # ====================================================
        # Positive consistency
        #
        # Equivalent facts + "remain consistent?"
        #
        # -> Yes
        # ====================================================

        if polarity == "positive":

            return RelationResolution(
                applied=
                    True,

                resolved_answer=
                    "Yes",

                relation_type=
                    relation_type,

                requested_polarity=
                    polarity,

                fact_count=
                    len(
                        facts
                    ),

                predicate_signatures=
                    signatures,

                reason=(
                    "Exact predicate equivalence "
                    "supports positive consistency."
                ),
            )


        # ====================================================
        # Explicit inconsistency
        #
        # Equivalent facts + "were they inconsistent?"
        #
        # -> No
        # ====================================================

        if polarity == "negative":

            return RelationResolution(
                applied=
                    True,

                resolved_answer=
                    "No",

                relation_type=
                    relation_type,

                requested_polarity=
                    polarity,

                fact_count=
                    len(
                        facts
                    ),

                predicate_signatures=
                    signatures,

                reason=(
                    "Exact predicate equivalence "
                    "rules out explicit "
                    "inconsistency."
                ),
            )


        # ====================================================
        # Defensive fallback
        # ====================================================

        return RelationResolution(
            applied=
                False,

            resolved_answer=
                None,

            relation_type=
                relation_type,

            requested_polarity=
                polarity,

            fact_count=
                len(
                    facts
                ),

            predicate_signatures=
                signatures,

            reason=(
                "Relation detected but no safe "
                "deterministic rule was available."
            ),
        )