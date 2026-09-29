import math
import re

from adaptive_agentic_rag.retrieval.query_decomposer import (
    QueryDecomposer,
)


DEFAULT_THRESHOLD = -6.127460241317749
DEFAULT_REQUIRED_FRACTION = 0.75


ABSENCE_PATTERNS = (
    r"\bwithout mentioning\b",
    r"\bwithout mention\b",
    r"\bdoes not mention\b",
    r"\bdoesn't mention\b",
    r"\bdid not mention\b",
    r"\bdidn't mention\b",
    r"\bwithout specifying\b",
    r"\bdoes not specify\b",
    r"\bdoesn't specify\b",
    r"\bdid not specify\b",
    r"\bdidn't specify\b",
    r"\bno mention of\b",
    r"\bno reference to\b",
)


MULTI_DOCUMENT_PATTERNS = (
    r"\bbetween\b",
    r"\bcompared to\b",
    r"\bcompared with\b",
    r"\bin contrast to\b",
    r"\bcontrast with\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bearlier report\b",
    r"\blater report\b",
    r"\bsubsequent report\b",
    r"\bsubsequent article\b",
    r"\bcontradict\b",
    r"\bcontradiction\b",
    r"\binconsisten",
    r"\bconsistent\b",
    r"\bboth articles\b",
    r"\bboth reports\b",
    r"\beach article\b",
    r"\beach report\b",
)


GENERIC_METADATA_TERMS = {
    "article",
    "articles",
    "report",
    "reports",
    "reported",
    "reporting",
    "source",
    "sources",
    "according",
    "covered",
    "covering",
    "published",
    "publication",
    "claim",
    "claims",
    "claimed",
}


SOURCE_GENERIC_TOKENS = {
    "the",
    "a",
    "an",
    "news",
    "world",
    "business",
    "leader",
    "online",
    "com",
}


PERSON_NONPERSON_TOKENS = {
    "ai",
    "openai",
    "chatgpt",
    "techcrunch",
    "verge",
    "bloomberg",
    "fortune",
    "guardian",
    "polygon",
    "journal",
    "times",
    "street",
    "news",
    "sports",
    "group",
    "company",
    "companies",
    "capital",
    "ventures",
    "venture",
    "research",
    "university",
    "institute",
    "technologies",
    "technology",
    "systems",
    "labs",
    "laboratories",
    "foundation",
    "association",
    "committee",
    "department",
    "government",
    "bank",
    "exchange",
}


PERSON_NAME_PATTERN = re.compile(
    r"\b("
    r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?"
    r"(?:\s+"
    r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?"
    r"){1,3}"
    r")\b"
)


SOURCE_CUE_PATTERNS = (
    re.compile(
        r"\b(?:as\s+)?"
        r"(?:reported|detailed|covered|described|published)"
        r"\s+by\s+"
        r"(?P<source>"
        r"(?:The\s+)?"
        r"[A-Z][A-Za-z0-9.&'-]*"
        r"(?:\s+[A-Z][A-Za-z0-9.&'-]*){0,5}?"
        r")"
        r"(?="
        r"\s+(?:and|while|which|that|who|where|when|article|report|update)\b"
        r"|[,.;?]"
        r"|$"
        r")"
    ),

    re.compile(
        r"\b(?:article|report|update)\s+"
        r"(?:from|by)\s+"
        r"(?P<source>"
        r"(?:The\s+)?"
        r"[A-Z][A-Za-z0-9.&'-]*"
        r"(?:\s+[A-Z][A-Za-z0-9.&'-]*){0,5}?"
        r")"
        r"(?="
        r"\s+(?:and|while|which|that|who|where|when)\b"
        r"|[,.;?]"
        r"|$"
        r")"
    ),

    re.compile(
        r"\baccording\s+to\s+"
        r"(?:(?:a|an|the|another)\s+)?"
        r"(?:(?:article|report|update)\s+)?"
        r"(?:by\s+)?"
        r"(?P<source>"
        r"(?:The\s+)?"
        r"[A-Z][A-Za-z0-9.&'-]*"
        r"(?:\s+[A-Z][A-Za-z0-9.&'-]*){0,5}?"
        r")"
        r"(?="
        r"\s+(?:and|while|which|that|who|where|when)\b"
        r"|[,.;?]"
        r"|$"
        r")"
    ),
)


class ConstrainedSemanticRescue:

    def __init__(
        self,
        reranker,
        evidence_grader,
        threshold: float = DEFAULT_THRESHOLD,
        required_fraction: float = DEFAULT_REQUIRED_FRACTION,
    ):

        self.reranker = reranker
        self.evidence_grader = evidence_grader

        self.threshold = float(
            threshold
        )

        self.required_fraction = float(
            required_fraction
        )

        self.decomposer = (
            QueryDecomposer()
        )


    # ========================================================
    # Normalization
    # ========================================================

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        return " ".join(
            (text or "").strip().split()
        )


    @staticmethod
    def _tokens(
        text: str,
    ) -> list[str]:

        return re.findall(
            r"[a-z0-9]+",
            (text or "").lower(),
        )


    @classmethod
    def _normalized_source(
        cls,
        source: str,
    ) -> str:

        return " ".join(
            cls._tokens(
                source
            )
        )


    @classmethod
    def _source_core_tokens(
        cls,
        source: str,
    ) -> list[str]:

        primary = (
            (source or "")
            .split("|", 1)[0]
            .strip()
        )


        tokens = (
            cls._tokens(
                primary
            )
        )


        filtered = [
            token

            for token
            in tokens

            if (
                token
                not in
                SOURCE_GENERIC_TOKENS
            )
        ]


        return (
            filtered
            or
            tokens
        )


    @classmethod
    def _source_signature(
        cls,
        source: str,
    ) -> str:

        primary = (
            (source or "")
            .split("|", 1)[0]
            .strip()
        )


        tokens = (
            cls._tokens(
                primary
            )
        )


        if (
            tokens
            and
            tokens[0]
            ==
            "the"
        ):

            tokens = (
                tokens[
                    1:
                ]
            )


        return "".join(
            tokens
        )


    # ========================================================
    # Requirements
    # ========================================================

    def extract_requirements(
        self,
        query: str,
    ) -> list[str]:

        output = (
            self.decomposer.decompose(
                query
            )
        )


        if not output:

            return [
                query
            ]


        if len(
            output
        ) > 1:

            requirements = (
                output[
                    1:
                ]
            )

        else:

            requirements = [
                output[
                    0
                ]
            ]


        cleaned = []

        seen = set()


        for requirement in requirements:

            normalized = (
                self._normalize(
                    requirement
                )
            )


            key = (
                normalized.lower()
            )


            if (
                not normalized
                or
                key in seen
            ):

                continue


            seen.add(
                key
            )


            cleaned.append(
                normalized
            )


        return (
            cleaned
            or
            [
                query
            ]
        )


    # ========================================================
    # Candidates
    # ========================================================

    @staticmethod
    def build_candidates(
        context,
    ) -> list[dict]:

        candidates = []


        for (
            index,
            item,
        ) in enumerate(
            context.items
        ):

            parts = []


            if item.source:

                parts.append(
                    f"Source: {item.source}"
                )


            if item.title:

                parts.append(
                    f"Title: {item.title}"
                )


            parts.append(
                f"Evidence: {item.text}"
            )


            candidates.append(
                {
                    "id":
                        f"context_{index}",

                    "text":
                        "\n".join(
                            parts
                        ),

                    "raw_text":
                        item.text or "",

                    "document_id":
                        item.document_id,

                    "chunk_id":
                        item.chunk_id,

                    "citation_id":
                        item.citation_id,

                    "source":
                        item.source or "",

                    "title":
                        item.title or "",
                }
            )


        return candidates


    # ========================================================
    # Source extraction
    # ========================================================

    def _candidate_source_is_mentioned(
        self,
        requirement: str,
        source: str,
    ) -> bool:

        requirement_tokens = set(
            self._tokens(
                requirement
            )
        )


        core_tokens = (
            self._source_core_tokens(
                source
            )
        )


        if (
            core_tokens
            and
            all(
                token
                in
                requirement_tokens

                for token
                in core_tokens
            )
        ):

            return True


        signature = (
            self._source_signature(
                source
            )
        )


        requirement_compact = "".join(
            self._tokens(
                requirement
            )
        )


        # Avoid very short/generic source signatures
        # such as "age".
        if (
            len(
                signature
            )
            >=
            6
            and
            signature
            in
            requirement_compact
        ):

            return True


        return False


    def _extract_source_phrases_from_text(
        self,
        text: str,
    ) -> list[str]:

        output = []


        # ----------------------------------------------------
        # Quoted publisher followed by article/report.
        # ----------------------------------------------------

        quoted_followed = re.compile(
            r"['\"]([^'\"]+)['\"]"
            r"\s+(?:article|report|update)\b"
        )


        for match in quoted_followed.finditer(
            text
        ):

            value = (
                self._normalize(
                    match.group(
                        1
                    )
                )
            )


            if value:

                output.append(
                    value
                )


        # ----------------------------------------------------
        # report/article from 'Publisher'
        # ----------------------------------------------------

        quoted_from = re.compile(
            r"\b(?:from|by)\s+"
            r"['\"]([^'\"]+)['\"]"
        )


        for match in quoted_from.finditer(
            text
        ):

            value = (
                self._normalize(
                    match.group(
                        1
                    )
                )
            )


            if value:

                output.append(
                    value
                )


        # ----------------------------------------------------
        # Reporting cues:
        #
        # reported by The Wall Street Journal
        # detailed by Bloomberg
        # covered by The Verge
        # report by TechCrunch
        # according to another report by TechCrunch
        # ----------------------------------------------------

        for pattern in (
            SOURCE_CUE_PATTERNS
        ):

            for match in pattern.finditer(
                text
            ):

                value = (
                    self._normalize(
                        match.group(
                            "source"
                        )
                    )
                )


                if value:

                    output.append(
                        value
                    )


        return output


    def explicit_sources(
        self,
        requirement: str,
        candidates: list[dict],
    ) -> list[str]:

        found = []


        unique_candidate_sources = list(
            dict.fromkeys(
                candidate[
                    "source"
                ]

                for candidate
                in candidates

                if candidate[
                    "source"
                ]
            )
        )


        # ----------------------------------------------------
        # Detect source names that actually exist in context.
        # ----------------------------------------------------

        for source in (
            unique_candidate_sources
        ):

            if (
                self._candidate_source_is_mentioned(
                    requirement=
                        requirement,

                    source=
                        source,
                )
            ):

                found.append(
                    source
                )


        # ----------------------------------------------------
        # Detect named sources even when they are missing from
        # context.
        #
        # This is essential for null questions.
        # ----------------------------------------------------

        found.extend(
            self._extract_source_phrases_from_text(
                requirement
            )
        )


        deduplicated = []

        seen = set()


        for source in found:

            signature = (
                self._source_signature(
                    source
                )
            )


            if (
                not signature
                or
                signature in seen
            ):

                continue


            seen.add(
                signature
            )


            deduplicated.append(
                source
            )


        return deduplicated


    def _source_matches(
        self,
        candidate_source: str,
        required_source: str,
    ) -> bool:

        candidate_signature = (
            self._source_signature(
                candidate_source
            )
        )


        required_signature = (
            self._source_signature(
                required_source
            )
        )


        if (
            not candidate_signature
            or
            not required_signature
        ):

            return False


        if (
            candidate_signature
            ==
            required_signature
        ):

            return True


        candidate_core = set(
            self._source_core_tokens(
                candidate_source
            )
        )


        required_core = set(
            self._source_core_tokens(
                required_source
            )
        )


        if (
            not candidate_core
            or
            not required_core
        ):

            return False


        return (
            required_core.issubset(
                candidate_core
            )
            or
            candidate_core.issubset(
                required_core
            )
        )


    def _missing_query_sources(
        self,
        query: str,
        candidates: list[dict],
    ) -> tuple[list[str], list[str]]:

        required_sources = (
            self.explicit_sources(
                requirement=
                    query,

                candidates=
                    candidates,
            )
        )


        missing = []


        for required_source in (
            required_sources
        ):

            exists = any(
                self._source_matches(
                    candidate_source=(
                        candidate[
                            "source"
                        ]
                    ),

                    required_source=(
                        required_source
                    ),
                )

                for candidate
                in candidates
            )


            if not exists:

                missing.append(
                    required_source
                )


        return (
            required_sources,
            missing,
        )


    # ========================================================
    # Critical anchors
    # ========================================================

    def _critical_terms(
        self,
        requirement: str,
    ) -> set[str]:

        return set(
            self.evidence_grader
            ._critical_terms(
                requirement
            )
        )


    def _local_critical_terms(
        self,
        requirement: str,
        explicit_sources: list[str],
    ) -> set[str]:

        critical_terms = (
            self._critical_terms(
                requirement
            )
        )


        removable = set(
            GENERIC_METADATA_TERMS
        )


        for source in (
            explicit_sources
        ):

            removable.update(
                self._tokens(
                    source
                )
            )


        return {
            term

            for term
            in critical_terms

            if (
                term.lower()
                not in
                removable
            )
        }


    def critical_coverage(
        self,
        text: str,
        critical_terms: set[str],
    ) -> float:

        if not critical_terms:

            return 1.0


        matched = {
            anchor

            for anchor
            in critical_terms

            if (
                self.evidence_grader
                ._anchor_present(
                    text,
                    anchor,
                )
            )
        }


        return (
            len(
                matched
            )
            /
            len(
                critical_terms
            )
        )


    # ========================================================
    # Absence claims
    # ========================================================

    @staticmethod
    def _is_absence_claim(
        requirement: str,
    ) -> bool:

        lowered = (
            requirement.lower()
        )


        return any(
            re.search(
                pattern,
                lowered,
            )

            for pattern
            in ABSENCE_PATTERNS
        )


    # ========================================================
    # Candidate eligibility
    # ========================================================

    def _eligible_candidates(
        self,
        requirement: str,
        candidates: list[dict],
        query_type: str,
    ) -> dict:

        explicit_sources = (
            self.explicit_sources(
                requirement=
                    requirement,

                candidates=
                    candidates,
            )
        )


        absence_claim = (
            self._is_absence_claim(
                requirement
            )
        )


        local_terms = (
            self._local_critical_terms(
                requirement=
                    requirement,

                explicit_sources=
                    explicit_sources,
            )
        )


        required_local_coverage = (
            self.evidence_grader
            ._critical_coverage_requirement(
                query_type,
                local_terms,
            )
        )


        multi_source_requirement = (
            len(
                explicit_sources
            )
            >
            1
        )


        if (
            absence_claim
            or
            multi_source_requirement
        ):

            return {
                "eligible":
                    [],

                "explicit_sources":
                    explicit_sources,

                "local_critical_terms":
                    sorted(
                        local_terms
                    ),

                "required_local_coverage":
                    required_local_coverage,

                "absence_claim":
                    absence_claim,

                "multi_source_requirement":
                    multi_source_requirement,
            }


        eligible = []


        for candidate in candidates:

            # ------------------------------------------------
            # Explicit source binding.
            # ------------------------------------------------

            if explicit_sources:

                source_ok = any(
                    self._source_matches(
                        candidate_source=(
                            candidate[
                                "source"
                            ]
                        ),

                        required_source=(
                            required_source
                        ),
                    )

                    for required_source
                    in explicit_sources
                )


                if not source_ok:

                    continue


            # ------------------------------------------------
            # Local anchor filtering happens BEFORE BGE.
            # ------------------------------------------------

            if local_terms:

                coverage = (
                    self.critical_coverage(
                        text=(
                            candidate[
                                "text"
                            ]
                        ),

                        critical_terms=
                            local_terms,
                    )
                )


                if (
                    coverage
                    <
                    required_local_coverage
                ):

                    continue


            else:

                coverage = 1.0


                # A source-only requirement is too weak.
                if explicit_sources:

                    continue


            eligible.append(
                {
                    **candidate,

                    "local_anchor_coverage":
                        coverage,
                }
            )


        return {
            "eligible":
                eligible,

            "explicit_sources":
                explicit_sources,

            "local_critical_terms":
                sorted(
                    local_terms
                ),

            "required_local_coverage":
                required_local_coverage,

            "absence_claim":
                absence_claim,

            "multi_source_requirement":
                multi_source_requirement,
        }


    # ========================================================
    # Requirement analysis
    # ========================================================

    def analyze_requirement(
        self,
        requirement: str,
        candidates: list[dict],
        query_type: str,
    ) -> dict:

        eligibility = (
            self._eligible_candidates(
                requirement=
                    requirement,

                candidates=
                    candidates,

                query_type=
                    query_type,
            )
        )


        eligible = (
            eligibility[
                "eligible"
            ]
        )


        base_result = {
            "text":
                requirement,

            "explicit_sources":
                eligibility[
                    "explicit_sources"
                ],

            "local_critical_terms":
                eligibility[
                    "local_critical_terms"
                ],

            "required_local_coverage":
                eligibility[
                    "required_local_coverage"
                ],

            "absence_claim":
                eligibility[
                    "absence_claim"
                ],

            "multi_source_requirement":
                eligibility[
                    "multi_source_requirement"
                ],

            "eligible_candidate_count":
                len(
                    eligible
                ),
        }


        if not eligible:

            return {
                **base_result,

                "supported":
                    False,

                "best_score":
                    None,

                "best_document_id":
                    None,

                "best_chunk_id":
                    None,

                "best_citation_id":
                    None,

                "best_source":
                    None,

                "best_title":
                    None,

                "best_local_anchor_coverage":
                    None,

                "best_raw_text":
                    None,
            }


        ranked = (
            self.reranker.rerank(
                query=
                    requirement,

                documents=
                    eligible,

                top_k=
                    len(
                        eligible
                    ),
            )
        )


        if not ranked:

            return {
                **base_result,

                "supported":
                    False,

                "best_score":
                    None,

                "best_document_id":
                    None,

                "best_chunk_id":
                    None,

                "best_citation_id":
                    None,

                "best_source":
                    None,

                "best_title":
                    None,

                "best_local_anchor_coverage":
                    None,

                "best_raw_text":
                    None,
            }


        best = ranked[
            0
        ]


        best_score = float(
            best[
                "rerank_score"
            ]
        )


        return {
            **base_result,

            "supported": (
                best_score
                >=
                self.threshold
            ),

            "best_score":
                best_score,

            "best_document_id":
                best[
                    "document_id"
                ],

            "best_chunk_id":
                best[
                    "chunk_id"
                ],

            "best_citation_id":
                best[
                    "citation_id"
                ],

            "best_source":
                best[
                    "source"
                ],

            "best_title":
                best[
                    "title"
                ],

            "best_local_anchor_coverage":
                best.get(
                    "local_anchor_coverage"
                ),

            "best_raw_text":
                best.get(
                    "raw_text",
                    "",
                ),
        }


    # ========================================================
    # Structural diversity
    # ========================================================

    @staticmethod
    def _requires_document_diversity(
        query: str,
        requirement_count: int,
    ) -> bool:

        if (
            requirement_count
            <
            2
        ):

            return False


        lowered = (
            query.lower()
        )


        return any(
            re.search(
                pattern,
                lowered,
            )

            for pattern
            in MULTI_DOCUMENT_PATTERNS
        )


    # ========================================================
    # Person bridge
    # ========================================================

    @staticmethod
    def _requires_person_bridge(
        query: str,
    ) -> bool:

        lowered = (
            " ".join(
                query.lower().split()
            )
        )


        return (
            lowered.startswith(
                "who "
            )
            or
            "which individual" in lowered
            or
            "which person" in lowered
            or
            "name of the individual" in lowered
            or
            "name of the person" in lowered
        )


    @staticmethod
    def _normalize_person_name(
        name: str,
    ) -> str:

        return " ".join(
            name.lower().split()
        )


    def _person_names(
        self,
        title: str,
        text: str,
    ) -> set[str]:

        combined = (
            f"{title or ''}\n"
            f"{text or ''}"
        )


        names = set()


        for match in (
            PERSON_NAME_PATTERN.finditer(
                combined
            )
        ):

            name = (
                self._normalize(
                    match.group(
                        1
                    )
                )
            )


            tokens = (
                self._tokens(
                    name
                )
            )


            if len(
                tokens
            ) < 2:

                continue


            if any(
                token
                in
                PERSON_NONPERSON_TOKENS

                for token
                in tokens
            ):

                continue


            names.add(
                self._normalize_person_name(
                    name
                )
            )


        return names


    def _person_bridge(
        self,
        query: str,
        supported_requirements: list[dict],
        required_count: int,
    ) -> dict:

        required = (
            self._requires_person_bridge(
                query
            )
        )


        if not required:

            return {
                "required":
                    False,

                "ok":
                    True,

                "candidate_people":
                    [],

                "per_requirement_people":
                    [],
            }


        per_requirement_people = []


        for requirement in (
            supported_requirements
        ):

            people = (
                self._person_names(
                    title=(
                        requirement.get(
                            "best_title"
                        )
                        or ""
                    ),

                    text=(
                        requirement.get(
                            "best_raw_text"
                        )
                        or ""
                    ),
                )
            )


            per_requirement_people.append(
                {
                    "requirement":
                        requirement[
                            "text"
                        ],

                    "document_id":
                        requirement[
                            "best_document_id"
                        ],

                    "people":
                        sorted(
                            people
                        ),
                }
            )


        counts = {}


        for item in (
            per_requirement_people
        ):

            # One person counts at most once
            # per requirement.
            for person in set(
                item[
                    "people"
                ]
            ):

                counts[
                    person
                ] = (
                    counts.get(
                        person,
                        0,
                    )
                    +
                    1
                )


        bridge_people = sorted(
            person

            for (
                person,
                count,
            ) in counts.items()

            if (
                count
                >=
                required_count
            )
        )


        return {
            "required":
                True,

            "ok":
                bool(
                    bridge_people
                ),

            "candidate_people":
                bridge_people,

            "per_requirement_people":
                per_requirement_people,
        }


    # ========================================================
    # Full rescue analysis
    # ========================================================

    def analyze(
        self,
        query: str,
        context,
        query_type: str,
    ) -> dict:

        candidates = (
            self.build_candidates(
                context
            )
        )


        (
            query_explicit_sources,
            missing_query_sources,
        ) = self._missing_query_sources(
            query=
                query,

            candidates=
                candidates,
        )


        query_source_coverage_ok = (
            len(
                missing_query_sources
            )
            ==
            0
        )


        requirements = (
            self.extract_requirements(
                query
            )
        )


        requirement_results = []


        for requirement in (
            requirements
        ):

            requirement_results.append(
                self.analyze_requirement(
                    requirement=
                        requirement,

                    candidates=
                        candidates,

                    query_type=
                        query_type,
                )
            )


        supported_requirements = [
            item

            for item
            in requirement_results

            if item[
                "supported"
            ]
        ]


        supported_count = len(
            supported_requirements
        )


        required_count = max(
            1,
            math.ceil(
                len(
                    requirement_results
                )
                *
                self.required_fraction
            ),
        )


        semantic_coverage_ok = (
            supported_count
            >=
            required_count
        )


        supporting_document_ids = list(
            dict.fromkeys(
                item[
                    "best_document_id"
                ]

                for item
                in supported_requirements

                if item[
                    "best_document_id"
                ]
            )
        )


        diversity_required = (
            self._requires_document_diversity(
                query=
                    query,

                requirement_count=
                    len(
                        requirement_results
                    ),
            )
        )


        if diversity_required:

            diversity_ok = (
                len(
                    supporting_document_ids
                )
                >=
                2
            )

        else:

            diversity_ok = True


        person_bridge = (
            self._person_bridge(
                query=
                    query,

                supported_requirements=
                    supported_requirements,

                required_count=
                    required_count,
            )
        )


        sufficient = (
            semantic_coverage_ok

            and
            query_source_coverage_ok

            and
            diversity_ok

            and
            person_bridge[
                "ok"
            ]
        )


        return {
            "sufficient":
                sufficient,

            "threshold":
                self.threshold,

            "required_fraction":
                self.required_fraction,

            # ------------------------------------------------
            # Query-level source completeness
            # ------------------------------------------------

            "query_explicit_sources":
                query_explicit_sources,

            "missing_query_sources":
                missing_query_sources,

            "query_source_coverage_ok":
                query_source_coverage_ok,

            # ------------------------------------------------
            # Requirements
            # ------------------------------------------------

            "requirement_count":
                len(
                    requirement_results
                ),

            "supported_requirement_count":
                supported_count,

            "required_requirement_count":
                required_count,

            "semantic_coverage_ok":
                semantic_coverage_ok,

            # ------------------------------------------------
            # Document diversity
            # ------------------------------------------------

            "document_diversity_required":
                diversity_required,

            "document_diversity_ok":
                diversity_ok,

            "supporting_document_ids":
                supporting_document_ids,

            # ------------------------------------------------
            # Entity bridge
            # ------------------------------------------------

            "person_bridge_required":
                person_bridge[
                    "required"
                ],

            "person_bridge_ok":
                person_bridge[
                    "ok"
                ],

            "person_bridge_candidates":
                person_bridge[
                    "candidate_people"
                ],

            "person_bridge_details":
                person_bridge[
                    "per_requirement_people"
                ],

            # ------------------------------------------------
            # Detailed requirement diagnostics
            # ------------------------------------------------

            "requirements":
                requirement_results,
        }