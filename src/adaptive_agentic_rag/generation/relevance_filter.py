import re

from dataclasses import dataclass


DEFAULT_MIN_RELEVANCE_SCORE = (
    -6.311376452445984
)


# ============================================================
# Public result objects
# ============================================================

@dataclass
class RelevantClaim:

    claim: str

    citation_id: int

    relevance_score: float


@dataclass
class RelevanceFilterResult:

    relevant_claims: list[RelevantClaim]

    filtered_claims: list[RelevantClaim]

    total_claims: int

    # --------------------------------------------------------
    # Selection telemetry.
    #
    # These fields make the adaptive behavior measurable
    # without changing the existing public claim lists.
    # --------------------------------------------------------

    selection_mode: str = "global_top_k"

    required_sources: list[str] | None = None

    covered_sources: list[str] | None = None

    adaptive_budget: int = 0


class ClaimRelevanceFilter:
    """
    Claim-level relevance selection.

    Production pipeline:

        supported claims
            ↓
        malformed-output guard
            ↓
        BGE cross-encoder ranking
            ↓
        calibrated catastrophic safety floor
            ↓
        source-aware adaptive selection
            ↓
        final relevant claims


    Important:

    DEFAULT_MIN_RELEVANCE_SCORE is a RAW BGE reranker logit.

    It is NOT a probability.

    The threshold is frozen from the previous calibration
    experiment and must not be retuned in this stage.
    """

    def __init__(
        self,
        reranker,
        max_relevant_claims: int = 2,
        max_adaptive_claims: int = 4,
        min_relevance_score: float = (
            DEFAULT_MIN_RELEVANCE_SCORE
        ),
    ):

        if (
            max_relevant_claims
            <
            1
        ):

            raise ValueError(
                "max_relevant_claims must be >= 1."
            )


        if (
            max_adaptive_claims
            <
            max_relevant_claims
        ):

            raise ValueError(
                (
                    "max_adaptive_claims must be >= "
                    "max_relevant_claims."
                )
            )


        self.reranker = (
            reranker
        )


        self.max_relevant_claims = (
            max_relevant_claims
        )


        self.max_adaptive_claims = (
            max_adaptive_claims
        )


        self.min_relevance_score = float(
            min_relevance_score
        )


    # ========================================================
    # Basic normalization
    # ========================================================

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        return " ".join(
            re.findall(
                r"[a-z0-9]+",
                (text or "").lower(),
            )
        )


    @staticmethod
    def _compact(
        text: str,
    ) -> str:

        return "".join(
            re.findall(
                r"[a-z0-9]+",
                (text or "").lower(),
            )
        )


    # ========================================================
    # Deterministic malformed-output guard
    # ========================================================

    @staticmethod
    def _is_malformed_claim(
        text: str,
    ) -> bool:

        normalized = (
            " ".join(
                (text or "").split()
            )
        )


        if not normalized:

            return True


        # ----------------------------------------------------
        # Placeholder / generation artifact.
        #
        # Examples:
        #
        # [Fact 1]
        # [TechCrunch article]
        #
        # Real prose may contain brackets internally, but an
        # entire bracket-wrapped claim is suspicious.
        # ----------------------------------------------------

        if re.fullmatch(
            r"\[[^\[\]]+\]",
            normalized,
        ):

            return True


        return False


    # ========================================================
    # Source normalization
    # ========================================================

    @classmethod
    def _canonical_source(
        cls,
        source: str,
    ) -> str:

        source = (
            source
            or ""
        ).strip()


        if not source:

            return ""


        # ----------------------------------------------------
        # Dataset source values sometimes contain:
        #
        # The Roar | Sports Writers Blog
        # CNBC | World Business News Leader
        #
        # Primary source identity is the left side.
        # ----------------------------------------------------

        primary = (
            source
            .split(
                "|",
                1,
            )[
                0
            ]
            .strip()
        )


        return cls._normalize(
            primary
        )


    @classmethod
    def _source_aliases(
        cls,
        source: str,
    ) -> set[str]:

        source = (
            source
            or ""
        ).strip()


        if not source:

            return set()


        primary = (
            source
            .split(
                "|",
                1,
            )[
                0
            ]
            .strip()
        )


        aliases = {
            cls._normalize(
                source
            ),

            cls._normalize(
                primary
            ),
        }


        expanded = set(
            aliases
        )


        for alias in aliases:

            # ------------------------------------------------
            # "The Sydney Morning Herald"
            #
            # may appear as:
            #
            # "Sydney Morning Herald"
            #
            # But do NOT collapse:
            #
            # "The Age" → "age"
            #
            # because that becomes dangerously generic.
            # ------------------------------------------------

            if alias.startswith(
                "the "
            ):

                without_the = (
                    alias[
                        4:
                    ]
                    .strip()
                )


                if (
                    len(
                        without_the.split()
                    )
                    >=
                    2
                ):

                    expanded.add(
                        without_the
                    )


        return {
            alias

            for alias
            in expanded

            if alias
        }


    # ========================================================
    # Does an alias occur in the query?
    # ========================================================

    @classmethod
    def _find_alias_position(
        cls,
        *,
        query: str,
        alias: str,
    ) -> int | None:

        query_normalized = (
            cls._normalize(
                query
            )
        )


        alias_normalized = (
            cls._normalize(
                alias
            )
        )


        if not alias_normalized:

            return None


        # ----------------------------------------------------
        # Normal word-boundary matching.
        # ----------------------------------------------------

        pattern = (
            r"(?:^|\s)"
            +
            re.escape(
                alias_normalized
            )
            +
            r"(?:$|\s)"
        )


        match = re.search(
            pattern,
            query_normalized,
        )


        if match:

            return (
                match.start()
            )


        # ----------------------------------------------------
        # Compact fallback:
        #
        # SportingNews
        #
        # vs
        #
        # Sporting News
        #
        # Only enable for reasonably long aliases to avoid
        # unsafe short substring matching.
        # ----------------------------------------------------

        compact_alias = (
            cls._compact(
                alias_normalized
            )
        )


        if (
            len(
                compact_alias
            )
            <
            6
        ):

            return None


        compact_query = (
            cls._compact(
                query
            )
        )


        position = (
            compact_query.find(
                compact_alias
            )
        )


        if (
            position
            ==
            -1
        ):

            return None


        return position


    # ========================================================
    # Sources explicitly referenced by query
    #
    # Returned in query order for deterministic selection.
    # ========================================================

    @classmethod
    def _mentioned_context_sources(
        cls,
        *,
        query: str,
        context,
    ) -> list[str]:

        if (
            context
            is None
        ):

            return []


        matches = {}


        for item in (
            getattr(
                context,
                "items",
                [],
            )
            or []
        ):

            source = (
                getattr(
                    item,
                    "source",
                    "",
                )
                or ""
            )


            canonical = (
                cls._canonical_source(
                    source
                )
            )


            if not canonical:

                continue


            best_position = None


            for alias in (
                cls._source_aliases(
                    source
                )
            ):

                position = (
                    cls._find_alias_position(
                        query=
                            query,

                        alias=
                            alias,
                    )
                )


                if (
                    position
                    is None
                ):

                    continue


                if (
                    best_position
                    is None
                    or
                    position
                    <
                    best_position
                ):

                    best_position = (
                        position
                    )


            if (
                best_position
                is None
            ):

                continue


            if (
                canonical
                not in
                matches
                or
                best_position
                <
                matches[
                    canonical
                ]
            ):

                matches[
                    canonical
                ] = (
                    best_position
                )


        return [
            source

            for (
                source,
                _
            )
            in sorted(
                matches.items(),
                key=lambda item:
                    item[
                        1
                    ],
            )
        ]


    # ========================================================
    # citation_id → canonical source
    # ========================================================

    @classmethod
    def _citation_source_map(
        cls,
        context,
    ) -> dict[int, str]:

        output = {}


        if (
            context
            is None
        ):

            return output


        for item in (
            getattr(
                context,
                "items",
                [],
            )
            or []
        ):

            citation_id = getattr(
                item,
                "citation_id",
                None,
            )


            if (
                citation_id
                is None
            ):

                continue


            source = (
                cls._canonical_source(
                    getattr(
                        item,
                        "source",
                        "",
                    )
                    or ""
                )
            )


            if source:

                output[
                    citation_id
                ] = (
                    source
                )


        return output


    # ========================================================
    # Convert ranked item to public claim result
    # ========================================================

    @staticmethod
    def _to_result(
        item,
    ) -> RelevantClaim:

        original_claim = (
            item[
                "claim"
            ]
        )


        return RelevantClaim(

            claim=(
                original_claim.claim
            ),

            citation_id=(
                original_claim.citation_id
            ),

            relevance_score=round(
                float(
                    item[
                        "rerank_score"
                    ]
                ),
                4,
            ),
        )


    # ========================================================
    # Global legacy selection
    # ========================================================

    def _select_global(
        self,
        survivors,
    ):

        selected = (
            survivors[
                :self.max_relevant_claims
            ]
        )


        rejected = (
            survivors[
                self.max_relevant_claims:
            ]
        )


        return (
            selected,
            rejected,
            self.max_relevant_claims,
        )


    # ========================================================
    # Source-aware adaptive selection
    # ========================================================

    def _select_source_aware(
        self,
        *,
        survivors,
        required_sources,
        citation_sources,
    ):

        # ----------------------------------------------------
        # Structural adaptive budget.
        #
        # Example:
        #
        # max_relevant_claims = 2
        # required sources    = 3
        #
        # adaptive budget     = 3
        #
        # No score threshold is changed.
        # ----------------------------------------------------

        budget = min(
            self.max_adaptive_claims,

            max(
                self.max_relevant_claims,
                len(
                    required_sources
                ),
            ),
        )


        selected = []

        selected_ids = set()


        # ====================================================
        # Pass 1:
        # Preserve the strongest surviving claim for each
        # explicitly required source.
        #
        # survivors are already sorted by BGE score.
        # ====================================================

        for source in (
            required_sources
        ):

            for index, item in enumerate(
                survivors
            ):

                claim = (
                    item[
                        "claim"
                    ]
                )


                citation_id = getattr(
                    claim,
                    "citation_id",
                    None,
                )


                claim_source = (
                    citation_sources.get(
                        citation_id,
                        "",
                    )
                )


                if (
                    claim_source
                    !=
                    source
                ):

                    continue


                if (
                    index
                    in
                    selected_ids
                ):

                    continue


                selected.append(
                    item
                )


                selected_ids.add(
                    index
                )


                break


        # ====================================================
        # Pass 2:
        # Fill remaining budget globally by relevance.
        # ====================================================

        for index, item in enumerate(
            survivors
        ):

            if (
                len(
                    selected
                )
                >=
                budget
            ):

                break


            if (
                index
                in
                selected_ids
            ):

                continue


            selected.append(
                item
            )


            selected_ids.add(
                index
            )


        # ----------------------------------------------------
        # Preserve global BGE ranking in final presentation.
        # ----------------------------------------------------

        selected.sort(
            key=lambda item:
                float(
                    item[
                        "rerank_score"
                    ]
                ),
            reverse=True,
        )


        rejected = [

            item

            for index, item
            in enumerate(
                survivors
            )

            if (
                index
                not in
                selected_ids
            )
        ]


        return (
            selected,
            rejected,
            budget,
        )


    # ========================================================
    # Main filter
    # ========================================================

    def filter(
        self,
        query: str,
        grounded_claims,
        context=None,
    ) -> RelevanceFilterResult:

        # ====================================================
        # Stage 0:
        # Grounder support is a hard prerequisite.
        # ====================================================

        supported_claims = [

            claim

            for claim
            in grounded_claims.claims

            if claim.supported
        ]


        if not supported_claims:

            return RelevanceFilterResult(
                relevant_claims=[],
                filtered_claims=[],
                total_claims=0,
                selection_mode=
                    "no_supported_claims",
                required_sources=[],
                covered_sources=[],
                adaptive_budget=0,
            )


        # ====================================================
        # Stage 1:
        # Remove deterministic malformed artifacts.
        # ====================================================

        valid_supported_claims = []

        malformed_results = []


        for claim in (
            supported_claims
        ):

            if self._is_malformed_claim(
                claim.claim
            ):

                malformed_results.append(
                    RelevantClaim(
                        claim=
                            claim.claim,

                        citation_id=
                            claim.citation_id,

                        relevance_score=
                            float(
                                "-inf"
                            ),
                    )
                )


                continue


            valid_supported_claims.append(
                claim
            )


        if not valid_supported_claims:

            return RelevanceFilterResult(
                relevant_claims=[],

                filtered_claims=
                    malformed_results,

                total_claims=len(
                    supported_claims
                ),

                selection_mode=
                    "malformed_only",

                required_sources=[],

                covered_sources=[],

                adaptive_budget=0,
            )


        # ====================================================
        # Stage 2:
        # Convert to BGEReranker format.
        # ====================================================

        documents = []


        for index, claim in enumerate(
            valid_supported_claims
        ):

            documents.append(
                {
                    "id":
                        f"claim_{index}",

                    "text":
                        claim.claim,

                    "claim":
                        claim,
                }
            )


        # ====================================================
        # Stage 3:
        # Global BGE ranking.
        # ====================================================

        ranked = (
            self.reranker.rerank(
                query=
                    query,

                documents=
                    documents,

                top_k=
                    len(
                        documents
                    ),
            )
        )


        # ====================================================
        # Stage 4:
        # Frozen catastrophic relevance floor.
        #
        # IMPORTANT:
        #
        # Source-aware selection is NOT allowed to rescue a
        # below-floor claim.
        # ====================================================

        survivors = []

        below_floor = []


        for item in (
            ranked
        ):

            score = float(
                item[
                    "rerank_score"
                ]
            )


            if (
                score
                <
                self.min_relevance_score
            ):

                below_floor.append(
                    item
                )

            else:

                survivors.append(
                    item
                )


        # ====================================================
        # Stage 5:
        # Detect explicitly mentioned sources.
        # ====================================================

        required_sources = (
            self._mentioned_context_sources(
                query=
                    query,

                context=
                    context,
            )
        )


        citation_sources = (
            self._citation_source_map(
                context
            )
        )


        # ====================================================
        # Stage 6:
        # Adaptive source-aware selection activates only for
        # genuine multi-source questions.
        #
        # One/no explicit source:
        # preserve old global top-k behavior.
        # ====================================================

        if (
            len(
                required_sources
            )
            >=
            2
        ):

            (
                selected,
                budget_rejected,
                adaptive_budget,
            ) = (
                self._select_source_aware(
                    survivors=
                        survivors,

                    required_sources=
                        required_sources,

                    citation_sources=
                        citation_sources,
                )
            )


            selection_mode = (
                "source_aware_adaptive"
            )


        else:

            (
                selected,
                budget_rejected,
                adaptive_budget,
            ) = (
                self._select_global(
                    survivors
                )
            )


            selection_mode = (
                "global_top_k"
            )


        # ====================================================
        # Stage 7:
        # Public result conversion.
        # ====================================================

        relevant_claims = [
            self._to_result(
                item
            )

            for item
            in selected
        ]


        filtered_claims = list(
            malformed_results
        )


        filtered_claims.extend(
            self._to_result(
                item
            )

            for item
            in below_floor
        )


        filtered_claims.extend(
            self._to_result(
                item
            )

            for item
            in budget_rejected
        )


        # ====================================================
        # Telemetry:
        # Which required sources actually survived selection?
        # ====================================================

        selected_citation_ids = {
            claim.citation_id

            for claim
            in relevant_claims

            if (
                claim.citation_id
                is not None
            )
        }


        covered_sources = []


        for source in (
            required_sources
        ):

            covered = any(
                (
                    citation_id
                    in
                    selected_citation_ids
                    and
                    citation_sources.get(
                        citation_id
                    )
                    ==
                    source
                )

                for citation_id
                in citation_sources
            )


            if covered:

                covered_sources.append(
                    source
                )


        return RelevanceFilterResult(
            relevant_claims=
                relevant_claims,

            filtered_claims=
                filtered_claims,

            total_claims=len(
                supported_claims
            ),

            selection_mode=
                selection_mode,

            required_sources=
                required_sources,

            covered_sources=
                covered_sources,

            adaptive_budget=
                adaptive_budget,
        )