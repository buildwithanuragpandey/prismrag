import re


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "both",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "would",
}


GENERIC_CAPITALIZED = (
    STOPWORDS
    |
    {
        "after",
        "before",
        "considering",
        "according",
        "following",
        "does",
        "which",
        "who",
        "what",
    }
)


class QueryDecomposer:
    """
    Deterministic query decomposition for multi-hop retrieval.

    The decomposer does NOT attempt to answer the question.

    Its purpose is to expose independent retrieval facets that
    may otherwise be diluted inside one very long multi-hop query.

    No LLM is used.
    """

    def __init__(
        self,
        max_facets: int = 3,
        extended_max_facets: int = 4,
        min_query_words: int = 18,
        extended_query_words: int = 45,
        min_facet_terms: int = 5
    ):

        #
        # Normal hard questions:
        #
        # original + up to 3 facets
        #

        self.max_facets = (
            max_facets
        )


        #
        # Very long multi-hop questions:
        #
        # original + up to 4 facets
        #

        self.extended_max_facets = (
            extended_max_facets
        )


        self.min_query_words = (
            min_query_words
        )


        self.extended_query_words = (
            extended_query_words
        )


        self.min_facet_terms = (
            min_facet_terms
        )


    # ========================================================
    # Normalize
    # ========================================================

    def _normalize(
        self,
        text: str
    ) -> str:

        return " ".join(
            text.strip().split()
        )


    # ========================================================
    # Content terms
    # ========================================================

    def _content_terms(
        self,
        text: str
    ) -> list[str]:

        tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )


        return [

            token

            for token in tokens

            if (
                len(token) > 1
                and
                token not in STOPWORDS
            )
        ]


    # ========================================================
    # Named entity / source-like signal
    # ========================================================

    def _has_named_signal(
        self,
        text: str
    ) -> bool:

        tokens = re.findall(
            r"\b[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*\b",
            text
        )


        for token in tokens:

            normalized = (
                token.lower()
            )


            if (
                normalized
                not in GENERIC_CAPITALIZED
            ):

                return True


        return False


    # ========================================================
    # Clean a clause
    # ========================================================

    def _clean_clause(
        self,
        clause: str
    ) -> str:

        clause = (
            self._normalize(
                clause
            )
        )


        clause = re.sub(
            r"^(?:and|but|while|whereas)\s+",
            "",
            clause,
            flags=re.IGNORECASE
        )


        return clause.strip(
            " ,;?."
        )


    # ========================================================
    # Split long query into candidate facets
    # ========================================================

    def _split_clauses(
        self,
        query: str
    ) -> list[str]:

        """
        MultiHopRAG frequently encodes individual evidence
        requirements as comma-separated clauses.

        We intentionally avoid globally splitting on "and".

        Example:

            Taylor Swift and Travis Kelce

        must remain together.
        """

        raw_parts = re.split(
            r"\s*;\s*|,\s*",
            query
        )


        cleaned = []


        for part in raw_parts:

            clause = (
                self._clean_clause(
                    part
                )
            )


            if clause:

                cleaned.append(
                    clause
                )


        return cleaned


    # ========================================================
    # Dynamic facet budget
    # ========================================================

    def _facet_limit(
        self,
        query: str
    ) -> int:

        word_count = len(
            query.split()
        )


        #
        # Monster queries can encode four or more
        # independent evidence requirements.
        #
        # Give them one extra retrieval facet.
        #

        if (
            word_count
            >=
            self.extended_query_words
        ):

            return (
                self.extended_max_facets
            )


        return (
            self.max_facets
        )


    # ========================================================
    # Decompose
    # ========================================================

    def decompose(
        self,
        query: str
    ) -> list[str]:

        original = (
            self._normalize(
                query
            )
        )


        if not original:

            return []


        word_count = len(
            original.split()
        )


        # ----------------------------------------------------
        # Short queries do not need decomposition.
        # ----------------------------------------------------

        if (
            word_count
            <
            self.min_query_words
        ):

            return [
                original
            ]


        facet_limit = (
            self._facet_limit(
                original
            )
        )


        clauses = (
            self._split_clauses(
                original
            )
        )


        facets = []

        pending_prefix = None


        # ====================================================
        # Build meaningful facets
        # ====================================================

        for clause in clauses:

            content_terms = (
                self._content_terms(
                    clause
                )
            )


            informative = (

                len(
                    content_terms
                )

                >=

                self.min_facet_terms
            )


            # ------------------------------------------------
            # Normal informative clause
            # ------------------------------------------------

            if informative:

                if pending_prefix:

                    facet = (
                        self._normalize(
                            f"{pending_prefix} {clause}"
                        )
                    )


                    pending_prefix = None


                else:

                    facet = (
                        clause
                    )


                facets.append(
                    facet
                )


                continue


            # ------------------------------------------------
            # Short source/entity clause
            #
            # Example:
            #
            # "as reported by Fortune"
            #
            # This should not become an independent retrieval
            # query, but its source signal is useful.
            # ------------------------------------------------

            if (
                self._has_named_signal(
                    clause
                )
            ):

                if facets:

                    facets[
                        -1
                    ] = (
                        self._normalize(

                            f"{facets[-1]} "
                            f"{clause}"
                        )
                    )


                else:

                    pending_prefix = (
                        clause
                    )


        # ====================================================
        # Deduplicate
        # ====================================================

        output = [
            original
        ]


        seen = {
            original.lower()
        }


        for facet in facets:

            normalized = (
                self._normalize(
                    facet
                )
            )


            key = (
                normalized.lower()
            )


            if not normalized:

                continue


            if key in seen:

                continue


            #
            # Reject very small fragments.
            #

            if (

                len(
                    self._content_terms(
                        normalized
                    )
                )

                <

                self.min_facet_terms
            ):

                continue


            output.append(
                normalized
            )


            seen.add(
                key
            )


            #
            # +1 because output[0] is always
            # the original query.
            #

            if (
                len(
                    output
                )
                >=
                facet_limit + 1
            ):

                break


        return output