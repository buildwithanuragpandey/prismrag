import re
from enum import Enum


class QueryType(str, Enum):
    SIMPLE = "simple"
    MULTIHOP = "multihop"
    COMPLEX = "complex"


class QueryRouter:

    def __init__(self):

        # High-confidence single-document lookup intents.
        self.simple_patterns = [
            re.compile(
                r"^who\s+(wrote|authored)\b",
                re.IGNORECASE
            ),
            re.compile(
                r"^when\s+was\b.*\bpublished\b",
                re.IGNORECASE
            ),
            re.compile(
                r"^which\s+(source|publication|website)\b.*\bpublished\b",
                re.IGNORECASE
            ),
        ]

        # Comparison / aggregation / broad-analysis intents.
        self.complex_patterns = [
            re.compile(
                r"\bcompare\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\bcomparison\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\bdifference(s)?\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\bversus\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\bvs\.?\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\bsummarize\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\banaly[sz]e\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\bmultiple\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\bseveral\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\bwhich one\b",
                re.IGNORECASE
            ),
        ]

        # Reasoning / multi-hop intents.
        self.multihop_patterns = [
            re.compile(
                r"\bwhy\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\bexplain\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\bhow does\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\brelationship\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\bbetween\b",
                re.IGNORECASE
            ),
        ]


    def _intent_text(
        self,
        query: str
    ) -> str:

        text = query.strip()

        # Remove quoted titles/entities before classification.
        #
        # Example:
        #
        # Who wrote the article
        # "Why Amazon Changed Everything"?
        #
        # We care about "Who wrote ...",
        # not the word "Why" inside the title.

        text = re.sub(
            r'"[^"]*"',
            '""',
            text
        )

        text = re.sub(
            r'“[^”]*”',
            '""',
            text
        )

        return " ".join(
            text.split()
        )


    def classify(
        self,
        query: str
    ) -> QueryType:

        intent_text = self._intent_text(
            query
        )

        # --------------------------------------------------
        # High-confidence simple lookup
        # --------------------------------------------------

        for pattern in self.simple_patterns:

            if pattern.search(
                intent_text
            ):
                return QueryType.SIMPLE

        # --------------------------------------------------
        # Comparison / aggregation
        # --------------------------------------------------

        for pattern in self.complex_patterns:

            if pattern.search(
                intent_text
            ):
                return QueryType.COMPLEX

        # --------------------------------------------------
        # Multi-hop reasoning
        # --------------------------------------------------

        for pattern in self.multihop_patterns:

            if pattern.search(
                intent_text
            ):
                return QueryType.MULTIHOP

        # --------------------------------------------------
        # Long free-form questions
        # --------------------------------------------------

        if len(
            intent_text.split()
        ) > 18:
            return QueryType.MULTIHOP

        # Cheap path by default.
        return QueryType.SIMPLE


    def route(
        self,
        query: str
    ):

        query_type = self.classify(
            query
        )

        if query_type == QueryType.SIMPLE:

            return {
                "query_type":
                    query_type.value,

                "retrieval_strategy":
                    "dense",

                "rerank":
                    False,

                "mmr":
                    False
            }

        # MULTIHOP and COMPLEX currently
        # share the heavy retrieval path.

        return {
            "query_type":
                query_type.value,

            "retrieval_strategy":
                "hybrid",

            "rerank":
                True,

            "mmr":
                True
        }
