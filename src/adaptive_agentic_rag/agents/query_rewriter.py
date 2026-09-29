import re


class QueryRewriter:
    """
    Deterministic retrieval-oriented query rewriter.

    V2 priority
    -----------
    When AdaptiveRetryPolicy approves a retry because an
    explicit source is missing, rewriting becomes:

        missing publisher
        +
        high-information query anchors

    rather than:

        original question + "supporting evidence"

    The rewriter remains:

    - deterministic
    - local
    - model-free
    - embedding-free
    - gold-label-free

    Legacy rewriting behavior remains available as fallback
    when no structural source telemetry is supplied.
    """

    # ========================================================
    # Retrieval-noise terms
    #
    # These are mostly question/reporting scaffolding.
    #
    # They are intentionally conservative: domain/content
    # terms should survive.
    # ========================================================

    STOPWORDS = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "which",
        "what",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "can",
        "could",
        "would",
        "should",
        "will",
        "may",
        "might",
        "must",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "with",
        "without",
        "into",
        "over",
        "under",
        "between",
        "among",
        "about",
        "after",
        "before",
        "during",
        "through",
        "according",
        "considering",
        "based",
        "while",
        "whereas",
        "also",
        "both",
        "another",
        "other",
        "same",
        "subsequent",
        "later",
        "earlier",
        "article",
        "articles",
        "report",
        "reports",
        "reported",
        "reporting",
        "coverage",
        "source",
        "sources",
        "news",
        "information",
        "evidence",
        "relevant",
        "supporting",
    }


    def __init__(
        self,
        *,
        max_target_terms: int = 34,
        local_window_chars: int = 240,
    ):

        self.max_target_terms = (
            max_target_terms
        )


        self.local_window_chars = (
            local_window_chars
        )


    # ========================================================
    # Normalization
    # ========================================================

    def _normalize(
        self,
        query: str,
    ) -> str:

        return " ".join(
            (
                query
                or ""
            )
            .strip()
            .split()
        )


    # ========================================================
    # Stable list normalization
    # ========================================================

    def _normalize_source_list(
        self,
        sources,
    ) -> list[str]:

        output = []

        seen = set()


        for source in (
            sources
            or []
        ):

            source = (
                self._normalize(
                    str(
                        source
                    )
                )
            )


            if not source:

                continue


            key = (
                source.lower()
            )


            if key in seen:

                continue


            seen.add(
                key
            )


            output.append(
                source
            )


        return output


    # ========================================================
    # Remove known source identities from content text
    #
    # We prefix the missing source explicitly, so repeating
    # source names inside retrieval anchors adds noise.
    # ========================================================

    def _remove_sources(
        self,
        text: str,
        sources,
    ) -> str:

        output = (
            text
        )


        ordered_sources = sorted(
            self._normalize_source_list(
                sources
            ),
            key=len,
            reverse=True,
        )


        for source in (
            ordered_sources
        ):

            output = re.sub(
                re.escape(
                    source
                ),
                " ",
                output,
                flags=re.IGNORECASE,
            )


        return self._normalize(
            output
        )


    # ========================================================
    # Local context around missing publisher
    # ========================================================

    def _local_source_context(
        self,
        *,
        query: str,
        source: str,
    ) -> str:

        match = re.search(
            re.escape(
                source
            ),
            query,
            flags=re.IGNORECASE,
        )


        if match is None:

            return ""


        start = max(
            0,
            (
                match.start()
                -
                self.local_window_chars
            ),
        )


        end = min(
            len(
                query
            ),
            (
                match.end()
                +
                self.local_window_chars
            ),
        )


        return self._normalize(
            query[
                start:
                end
            ]
        )


    # ========================================================
    # Retrieval tokens
    # ========================================================

    @staticmethod
    def _tokens(
        text: str,
    ) -> list[str]:

        return re.findall(
            (
                r"[A-Za-z0-9]"
                r"[A-Za-z0-9"
                r".&'’/_+-]*"
            ),
            text or "",
        )


    # ========================================================
    # High-information anchors
    # ========================================================

    def _content_terms(
        self,
        text: str,
        *,
        limit: int,
    ) -> list[str]:

        output = []

        seen = set()


        for token in (
            self._tokens(
                text
            )
        ):

            cleaned = (
                token.strip(
                    ".,;:!?()[]{}"
                    "\"'“”‘’"
                )
            )


            if not cleaned:

                continue


            key = (
                cleaned.lower()
            )


            if key in (
                self.STOPWORDS
            ):

                continue


            # -----------------------------------------------
            # Ignore isolated punctuation-like fragments.
            # -----------------------------------------------

            if (
                len(
                    cleaned
                )
                ==
                1
                and
                not cleaned.isdigit()
            ):

                continue


            if key in seen:

                continue


            seen.add(
                key
            )


            output.append(
                cleaned
            )


            if (
                len(
                    output
                )
                >=
                limit
            ):

                break


        return output


    # ========================================================
    # Missing-source targeted rewrite
    # ========================================================

    def _rewrite_missing_sources(
        self,
        *,
        query: str,
        required_sources: list[str],
        covered_sources: list[str],
        missing_sources: list[str],
    ) -> str:

        missing_sources = (
            self._normalize_source_list(
                missing_sources
            )
        )


        covered_sources = (
            self._normalize_source_list(
                covered_sources
            )
        )


        required_sources = (
            self._normalize_source_list(
                required_sources
            )
        )


        if not missing_sources:

            return ""


        # ====================================================
        # Prefix is the retrieval target.
        # ====================================================

        source_prefix = " ".join(
            missing_sources
        )


        # ====================================================
        # Remove all known publisher identities before
        # extracting semantic anchors.
        #
        # This prevents already-covered publishers from
        # dominating retry retrieval.
        # ====================================================

        all_sources = (
            required_sources
            +
            covered_sources
            +
            missing_sources
        )


        global_content = (
            self._remove_sources(
                query,
                all_sources,
            )
        )


        # ====================================================
        # First collect local anchors around each missing
        # publisher.
        #
        # These often contain:
        #
        # - dates
        # - local entities
        # - event descriptors
        # - article-specific topic
        # ====================================================

        local_terms = []

        local_seen = set()


        for source in (
            missing_sources
        ):

            local_context = (
                self._local_source_context(
                    query=
                        query,

                    source=
                        source,
                )
            )


            local_context = (
                self._remove_sources(
                    local_context,
                    all_sources,
                )
            )


            terms = (
                self._content_terms(
                    local_context,
                    limit=(
                        self.max_target_terms
                    ),
                )
            )


            for term in terms:

                key = (
                    term.lower()
                )


                if key in local_seen:

                    continue


                local_seen.add(
                    key
                )


                local_terms.append(
                    term
                )


        # ====================================================
        # Supplement with whole-query anchors.
        #
        # Important for queries where the source is mentioned
        # early but source-specific content occurs later.
        #
        # Example:
        #
        # CBSSports.com is mentioned in the opening clause,
        # while Josh Dobbs / Kirk Cousins appear later.
        # ====================================================

        global_terms = (
            self._content_terms(
                global_content,
                limit=(
                    self.max_target_terms
                    *
                    2
                ),
            )
        )


        combined_terms = []

        seen_terms = set()


        for term in (
            local_terms
            +
            global_terms
        ):

            key = (
                term.lower()
            )


            if key in seen_terms:

                continue


            seen_terms.add(
                key
            )


            combined_terms.append(
                term
            )


            if (
                len(
                    combined_terms
                )
                >=
                self.max_target_terms
            ):

                break


        # ====================================================
        # Defensive fallback.
        # ====================================================

        if not combined_terms:

            return self._normalize(
                source_prefix
            )


        return self._normalize(
            (
                source_prefix
                +
                " "
                +
                " ".join(
                    combined_terms
                )
            )
        )


    # ========================================================
    # Legacy comparison rewrite
    # ========================================================

    def _rewrite_comparison(
        self,
        query: str,
    ) -> str:

        text = (
            self._normalize(
                query
            )
        )


        text = re.sub(
            r"^\s*compare\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )


        text = re.sub(
            (
                r"^\s*comparison\s+"
                r"(?:of|between)\s+"
            ),
            "",
            text,
            flags=re.IGNORECASE,
        )


        if not re.search(
            r"\bcomparison\b",
            text,
            flags=re.IGNORECASE,
        ):

            text = (
                text.rstrip(
                    " ?."
                )
                +
                " comparison"
            )


        return self._normalize(
            text
        )


    # ========================================================
    # Legacy reasoning rewrite
    # ========================================================

    def _rewrite_reasoning(
        self,
        query: str,
    ) -> str:

        text = (
            self._normalize(
                query
            )
        )


        if re.match(
            r"^\s*why\b",
            text,
            flags=re.IGNORECASE,
        ):

            text = re.sub(
                r"^\s*why\b",
                "",
                text,
                count=1,
                flags=re.IGNORECASE,
            )


            return self._normalize(
                f"{text} reasons explanation"
            )


        if re.match(
            r"^\s*how\b",
            text,
            flags=re.IGNORECASE,
        ):

            return self._normalize(
                f"{text} explanation"
            )


        return self._normalize(
            f"{text} supporting evidence"
        )


    # ========================================================
    # Legacy complex rewrite
    # ========================================================

    def _rewrite_complex(
        self,
        query: str,
    ) -> str:

        text = (
            self._normalize(
                query
            )
        )


        if re.search(
            (
                r"\bcompare\b|"
                r"\bcomparison\b|"
                r"\bversus\b|"
                r"\bvs\.?\b"
            ),
            text,
            flags=re.IGNORECASE,
        ):

            return self._rewrite_comparison(
                text
            )


        return self._normalize(
            f"{text} relevant facts evidence"
        )


    # ========================================================
    # Public rewrite
    # ========================================================

    def rewrite(
        self,
        query: str,
        query_type: str,
        attempt: int = 1,
        *,
        required_sources: list[str] | None = None,
        covered_sources: list[str] | None = None,
        missing_sources: list[str] | None = None,
    ) -> str:

        text = (
            self._normalize(
                query
            )
        )


        if not text:

            return text


        required_sources = (
            self._normalize_source_list(
                required_sources
            )
        )


        covered_sources = (
            self._normalize_source_list(
                covered_sources
            )
        )


        missing_sources = (
            self._normalize_source_list(
                missing_sources
            )
        )


        # ====================================================
        # V2 structural retry path
        #
        # Missing-source telemetry has priority over generic
        # query-type rewriting.
        # ====================================================

        if missing_sources:

            targeted = (
                self._rewrite_missing_sources(
                    query=
                        text,

                    required_sources=
                        required_sources,

                    covered_sources=
                        covered_sources,

                    missing_sources=
                        missing_sources,
                )
            )


            if targeted:

                return targeted


        # ====================================================
        # Legacy fallback
        #
        # Preserves old behavior for callers which are not
        # executing an AdaptiveRetryPolicy source-miss path.
        # ====================================================

        query_type = (
            query_type
            or ""
        ).strip().lower()


        if re.search(
            (
                r"^\s*compare\b|"
                r"\bcomparison\b|"
                r"\bversus\b|"
                r"\bvs\.?\b"
            ),
            text,
            flags=re.IGNORECASE,
        ):

            rewritten = (
                self._rewrite_comparison(
                    text
                )
            )


        elif (
            query_type
            ==
            "complex"
        ):

            rewritten = (
                self._rewrite_complex(
                    text
                )
            )


        elif (
            query_type
            ==
            "multihop"
        ):

            rewritten = (
                self._rewrite_reasoning(
                    text
                )
            )


        else:

            rewritten = (
                text
            )


        if (
            attempt
            >
            1
            and
            rewritten
            ==
            text
        ):

            rewritten = (
                self._normalize(
                    (
                        f"{rewritten} "
                        "relevant information"
                    )
                )
            )


        return rewritten