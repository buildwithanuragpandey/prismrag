import re

from dataclasses import dataclass

from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext
)


# ============================================================
# Generic lexical stopwords
# ============================================================

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
    "i",
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


# ============================================================
# Tokens that may be capitalized because of grammar,
# generic question wording, dates, or relational phrasing.
#
# They must NOT become hard evidence anchors.
# ============================================================

CRITICAL_TERM_EXCLUSIONS = (

    STOPWORDS

    |

    {
        # ----------------------------------------------------
        # Relational / question wording
        # ----------------------------------------------------

        "after",
        "before",
        "between",
        "both",
        "considering",
        "according",
        "based",
        "following",
        "given",

        "first",
        "second",
        "third",
        "other",
        "another",
        "additional",

        # ----------------------------------------------------
        # Generic source / reporting vocabulary
        # ----------------------------------------------------

        "article",
        "articles",
        "report",
        "reports",
        "reported",
        "reporting",
        "published",
        "source",
        "sources",
        "news",

        # ----------------------------------------------------
        # Generic comparison wording
        # ----------------------------------------------------

        "compare",
        "compared",
        "comparison",

        # ----------------------------------------------------
        # Generic answer-target wording
        # ----------------------------------------------------

        "individual",
        "person",
        "company",
        "organization",

        # ----------------------------------------------------
        # Generic legal wording
        # ----------------------------------------------------

        "case",
        "trial",
        "jury",

        # ----------------------------------------------------
        # Previously observed false anchors
        # ----------------------------------------------------

        "age",
        "life",
        "style",

        # ----------------------------------------------------
        # Boolean answer words
        # ----------------------------------------------------

        "yes",
        "no",

        # ----------------------------------------------------
        # Months
        # ----------------------------------------------------

        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }

)


# ============================================================
# Result model
# ============================================================

@dataclass
class EvidenceGrade:

    sufficient: bool

    evidence_score: float

    query_term_coverage: float

    unique_documents: int

    chunk_count: int

    weak_citations: list[int]

    reasons: list[str]


# ============================================================
# Evidence grader
# ============================================================

class EvidenceGrader:


    def __init__(
        self,
        weak_item_threshold: float = 0.20,
        evidence_score_threshold: float =  0.78,
        hard_critical_coverage: float = 0.75,
        simple_critical_coverage: float = 0.50
    ):

        self.weak_item_threshold = (
            weak_item_threshold
        )

        self.evidence_score_threshold = (
            evidence_score_threshold
        )

        self.hard_critical_coverage = (
            hard_critical_coverage
        )

        self.simple_critical_coverage = (
            simple_critical_coverage
        )


    # ========================================================
    # General tokenization
    # ========================================================

    def _tokenize(
        self,
        text: str
    ) -> set[str]:

        tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            (
                text
                or ""
            ).lower()
        )


        return {

            token

            for token in tokens

            if (
                token not in STOPWORDS
                and
                len(token) > 1
            )
        }


    def _query_terms(
        self,
        query: str
    ) -> set[str]:

        return self._tokenize(
            query
        )


    # ========================================================
    # Critical anchor normalization
    # ========================================================

    @staticmethod
    def _normalize_critical_token(
        token: str
    ) -> str:

        normalized = (
            token
            .replace(
                "’",
                "'"
            )
            .lower()
            .strip()
        )


        # ----------------------------------------------------
        # Normalize English possessives
        #
        # Google's
        #     -> google
        #
        # Bankman-Fried's
        #     -> bankman-fried
        #
        # publishers'
        #     -> publishers
        # ----------------------------------------------------

        normalized = re.sub(
            r"(?:'s|s')$",
            "",
            normalized
        )


        normalized = (
            normalized
            .strip(
                "-'"
            )
        )


        return normalized


    @staticmethod
    def _anchor_parts(
        anchor: str
    ) -> list[str]:

        return re.findall(
            r"[a-z0-9]+",
            (
                anchor
                or ""
            ).lower()
        )


    def _anchor_present(
        self,
        text: str,
        anchor: str
    ) -> bool:

        parts = (
            self._anchor_parts(
                anchor
            )
        )


        if not parts:

            return False


        # ----------------------------------------------------
        # Single-token anchor
        #
        # google
        # techcrunch
        # acmemart
        # ----------------------------------------------------

        if len(
            parts
        ) == 1:

            pattern = (

                r"\b"

                +

                re.escape(
                    parts[0]
                )

                +

                r"\b"
            )


        # ----------------------------------------------------
        # Compound anchor
        #
        # bankman-fried
        #
        # Accept:
        #
        # Bankman-Fried
        # Bankman Fried
        # Bankman'Fried
        # ----------------------------------------------------

        else:

            separator = (
                r"(?:[-'\s]+)"
            )


            pattern = (

                r"\b"

                +

                separator.join(

                    re.escape(
                        part
                    )

                    for part
                    in parts
                )

                +

                r"\b"
            )


        return bool(

            re.search(

                pattern,

                (
                    text
                    or ""
                ).lower()
            )
        )


    # ========================================================
    # Critical evidence anchors
    #
    # These remain a HARD safety signal.
    #
    # Examples:
    #
    # Amazon
    # AcmeMart
    # Taylor
    # Swift
    # Travis
    # Kelce
    # TechCrunch
    # Google
    # YouTube
    # Bankman-Fried
    #
    # Unlike the previous implementation, query anchors and
    # context anchors now use compatible normalization.
    # ========================================================

    def _critical_terms(
        self,
        query: str
    ) -> set[str]:

        raw_tokens = re.findall(

            (
                r"\b"
                r"[A-Z][A-Za-z0-9]*"
                r"(?:[-'][A-Za-z0-9]+)*"
                r"\b"
            ),

            query
            or ""
        )


        critical_terms = set()


        for token in raw_tokens:

            normalized = (
                self._normalize_critical_token(
                    token
                )
            )


            if not normalized:

                continue


            parts = (
                self._anchor_parts(
                    normalized
                )
            )


            if not parts:

                continue


            # ------------------------------------------------
            # Ignore anchors that collapse entirely to
            # single-character fragments.
            # ------------------------------------------------

            if all(
                len(part) <= 1
                for part in parts
            ):

                continue


            # ------------------------------------------------
            # Generic capitalized words should not become
            # hard entity anchors.
            # ------------------------------------------------

            if (
                normalized
                in
                CRITICAL_TERM_EXCLUSIONS
            ):

                continue


            if (
                len(parts) == 1
                and
                parts[0]
                in
                CRITICAL_TERM_EXCLUSIONS
            ):

                continue


            critical_terms.add(
                normalized
            )


        return critical_terms


    # ========================================================
    # Coverage
    # ========================================================

    def _coverage(
        self,
        query_terms: set[str],
        text: str
    ) -> float:

        if not query_terms:

            return 1.0


        text_terms = (
            self._tokenize(
                text
            )
        )


        matched_terms = (
            query_terms
            &
            text_terms
        )


        return (

            len(
                matched_terms
            )

            /

            len(
                query_terms
            )
        )


    # ========================================================
    # Evidence requirements
    # ========================================================

    def _requirements(
        self,
        query_type: str
    ) -> tuple[int, int, float]:

        """
        Returns:

        required_documents,
        required_chunks,
        preferred_query_coverage

        Document/chunk requirements remain HARD.

        Query-term coverage is now a SOFT signal:
        it contributes strongly to evidence_score but can no
        longer veto otherwise strong evidence by itself.
        """


        if query_type == "simple":

            return (
                1,
                1,
                0.50
            )


        if query_type in {
            "multihop",
            "complex"
        }:

            return (
                2,
                2,
                0.60
            )


        # Conservative fallback.

        return (
            2,
            2,
            0.60
        )


    # ========================================================
    # Critical-anchor threshold
    # ========================================================

    def _critical_coverage_requirement(
        self,
        query_type: str,
        critical_terms: set[str]
    ) -> float:

        if not critical_terms:

            return 0.0


        # ----------------------------------------------------
        # One critical named anchor must be present.
        # ----------------------------------------------------

        if len(
            critical_terms
        ) == 1:

            return 1.0


        if query_type == "simple":

            return (
                self.simple_critical_coverage
            )


        return (
            self.hard_critical_coverage
        )


    # ========================================================
    # Main grading
    # ========================================================

    def grade(
        self,
        query: str,
        context: BuiltContext,
        query_type: str
    ) -> EvidenceGrade:


        query_terms = (
            self._query_terms(
                query
            )
        )


        critical_terms = (
            self._critical_terms(
                query
            )
        )


        required_documents, (
            required_chunks
        ), preferred_coverage = (
            self._requirements(
                query_type
            )
        )


        # ====================================================
        # Empty context
        # ====================================================

        if not context.items:

            return EvidenceGrade(

                sufficient=False,

                evidence_score=0.0,

                query_term_coverage=0.0,

                unique_documents=0,

                chunk_count=0,

                weak_citations=[],

                reasons=[
                    "No evidence was retrieved."
                ]
            )


        # ====================================================
        # Overall lexical coverage
        #
        # IMPORTANT:
        #
        # This remains part of evidence_score.
        #
        # It is no longer a standalone hard veto.
        # ====================================================

        overall_coverage = (
            self._coverage(
                query_terms,
                context.text
            )
        )


        coverage_below_preferred = (

            overall_coverage

            <

            preferred_coverage
        )


        # ====================================================
        # Critical anchor coverage
        #
        # Unlike broad term coverage, named anchors remain
        # a hard safety gate.
        # ====================================================

        if critical_terms:

            matched_critical_terms = {

                anchor

                for anchor
                in critical_terms

                if self._anchor_present(
                    context.text,
                    anchor
                )
            }


            missing_critical_terms = (

                critical_terms

                -

                matched_critical_terms
            )


            critical_coverage = (

                len(
                    matched_critical_terms
                )

                /

                len(
                    critical_terms
                )
            )


        else:

            matched_critical_terms = set()

            missing_critical_terms = set()

            critical_coverage = 1.0


        minimum_critical_coverage = (
            self._critical_coverage_requirement(
                query_type,
                critical_terms
            )
        )


        # ====================================================
        # Document diversity
        # ====================================================

        unique_document_ids = {

            item.document_id

            for item
            in context.items
        }


        unique_documents = len(
            unique_document_ids
        )


        chunk_count = len(
            context.items
        )


        # ====================================================
        # Detect weak individual evidence
        #
        # Diagnostic only.
        # It does not veto an answer by itself.
        # ====================================================

        weak_citations = []


        for item in context.items:

            item_coverage = (
                self._coverage(
                    query_terms,
                    item.text
                )
            )


            if (
                item_coverage
                <
                self.weak_item_threshold
            ):

                weak_citations.append(
                    item.citation_id
                )


        # ====================================================
        # Component scores
        # ====================================================

        document_score = min(

            unique_documents
            /
            required_documents,

            1.0
        )


        chunk_score = min(

            chunk_count
            /
            required_chunks,

            1.0
        )


        # ====================================================
        # Combined score
        #
        # IMPORTANT:
        #
        # Keep the old weighting unchanged for this
        # controlled experiment.
        #
        # Broad lexical coverage still matters heavily,
        # but continuously rather than as a brittle veto.
        #
        # Critical anchors remain outside this score because
        # they are a separate hard safety condition.
        # ====================================================

        evidence_score = (

            0.55
            *
            overall_coverage

            +

            0.30
            *
            document_score

            +

            0.15
            *
            chunk_score
        )


        # ====================================================
        # Hard requirements
        # ====================================================

        enough_documents = (

            unique_documents

            >=

            required_documents
        )


        enough_chunks = (

            chunk_count

            >=

            required_chunks
        )


        enough_critical_coverage = (

            critical_coverage

            >=

            minimum_critical_coverage
        )


        enough_evidence_score = (

            evidence_score

            >=

            self.evidence_score_threshold
        )


        # ====================================================
        # Final decision
        #
        # Notice:
        #
        # overall_coverage >= preferred_coverage
        #
        # is intentionally NOT here anymore.
        # ====================================================

        sufficient = (

            enough_documents

            and

            enough_chunks

            and

            enough_critical_coverage

            and

            enough_evidence_score
        )


        # ====================================================
        # Human-readable reasons
        # ====================================================

        reasons = []


        if not enough_documents:

            reasons.append(

                (
                    "Insufficient document diversity: "
                    f"{unique_documents}/"
                    f"{required_documents}"
                )
            )


        if not enough_chunks:

            reasons.append(

                (
                    "Too few evidence chunks: "
                    f"{chunk_count}/"
                    f"{required_chunks}"
                )
            )


        # ----------------------------------------------------
        # Preserve the old prefix because our evaluation
        # parser already knows how to extract the numbers.
        #
        # It is now explicitly diagnostic only.
        # ----------------------------------------------------

        if coverage_below_preferred:

            reasons.append(

                (
                    "Query term coverage is too low: "
                    f"{overall_coverage:.2f} "
                    f"< {preferred_coverage:.2f} "
                    "(soft signal only)"
                )
            )


        if not enough_critical_coverage:

            missing_display = sorted(
                missing_critical_terms
            )


            reasons.append(

                (
                    "Critical query anchors are missing "
                    "from the evidence: "
                    f"coverage={critical_coverage:.2f} "
                    f"< {minimum_critical_coverage:.2f}; "
                    f"missing={missing_display}"
                )
            )


        if not enough_evidence_score:

            reasons.append(

                (
                    "Evidence score is too low: "
                    f"{evidence_score:.2f} "
                    f"< "
                    f"{self.evidence_score_threshold:.2f}"
                )
            )


        if weak_citations:

            reasons.append(

                (
                    "Potentially weak evidence citations: "
                    f"{weak_citations}"
                )
            )


        if sufficient:

            reasons.insert(

                0,

                "Evidence appears sufficient."
            )


        return EvidenceGrade(

            sufficient=sufficient,

            evidence_score=round(
                evidence_score,
                4
            ),

            query_term_coverage=round(
                overall_coverage,
                4
            ),

            unique_documents=(
                unique_documents
            ),

            chunk_count=chunk_count,

            weak_citations=(
                weak_citations
            ),

            reasons=reasons
        )