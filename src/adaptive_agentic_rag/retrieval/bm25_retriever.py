import json
import re

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Lexical retriever over processed chunks.

    Index representation includes:

    - chunk text
    - document title
    - source name

    Production extension
    --------------------
    search_by_sources() provides a source-constrained
    candidate-generation path for structurally approved
    retrieval retries.

    Normal search() behavior is unchanged.
    """

    def __init__(
        self,
        corpus_path: str = (
            "data/processed/"
            "processed_corpus_v2.json"
        ),
    ):

        with open(
            corpus_path,
            encoding="utf-8",
        ) as f:

            self.documents = json.load(
                f
            )


        self.texts = [
            self._document_tokens(
                document
            )

            for document
            in self.documents
        ]


        self.bm25 = BM25Okapi(
            self.texts
        )


        self.doc_map = {
            document["id"]:
                document

            for document
            in self.documents
        }


    # ========================================================
    # Normalize/tokenize
    # ========================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:

        if not text:

            return []


        return re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )


    # ========================================================
    # Source normalization
    # ========================================================

    @staticmethod
    def _normalize_source(
        source: str,
    ) -> str:

        return " ".join(
            re.findall(
                r"[a-z0-9]+",
                (
                    source
                    or ""
                ).lower(),
            )
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
            )[0]
            .strip()
        )


        aliases = {
            cls._normalize_source(
                source
            ),

            cls._normalize_source(
                primary
            ),
        }


        expanded = set(
            aliases
        )


        for alias in aliases:

            if not alias.startswith(
                "the "
            ):

                continue


            without_the = (
                alias[
                    4:
                ]
                .strip()
            )


            words = (
                without_the.split()
            )


            # The New York Times
            # -> New York Times

            if (
                len(
                    words
                )
                >=
                2
            ):

                expanded.add(
                    without_the
                )


            # The Guardian -> Guardian
            # The Verge    -> Verge
            #
            # But:
            #
            # The Age -> no unsafe "Age" alias.

            elif (
                len(
                    words
                )
                ==
                1
                and
                len(
                    words[
                        0
                    ]
                )
                >=
                5
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


    @classmethod
    def _source_matches(
        cls,
        *,
        actual_source: str,
        requested_source: str,
    ) -> bool:

        requested = (
            cls._normalize_source(
                requested_source
            )
        )


        if not requested:

            return False


        return (
            requested
            in
            cls._source_aliases(
                actual_source
            )
        )


    # ========================================================
    # Searchable document representation
    # ========================================================

    def _document_tokens(
        self,
        document: dict,
    ) -> list[str]:

        metadata = (
            document.get(
                "metadata",
                {},
            )
            or {}
        )


        title = (
            metadata.get(
                "title",
                "",
            )
        )


        source = (
            metadata.get(
                "source",
                "",
            )
        )


        text = (
            document.get(
                "text",
                "",
            )
        )


        searchable_text = (
            f"{title}\n"
            f"{source}\n"
            f"{text}"
        )


        return self._tokenize(
            searchable_text
        )


    # ========================================================
    # Public-result representation
    # ========================================================

    @staticmethod
    def _result_from_document(
        *,
        document: dict,
        score: float,
    ) -> dict:

        return {
            "id":
                document[
                    "id"
                ],

            "document_id":
                document[
                    "document_id"
                ],

            "text":
                document[
                    "text"
                ],

            "metadata":
                document.get(
                    "metadata",
                    {},
                ),

            "score":
                float(
                    score
                ),
        }


    # ========================================================
    # Normal BM25 search
    #
    # Existing production behavior.
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[dict]:

        if (
            not query
            or
            top_k <= 0
        ):

            return []


        query_tokens = (
            self._tokenize(
                query
            )
        )


        if not query_tokens:

            return []


        scores = (
            self.bm25.get_scores(
                query_tokens
            )
        )


        ranked = sorted(
            enumerate(
                scores
            ),
            key=lambda item: (
                -float(
                    item[
                        1
                    ]
                ),
                item[
                    0
                ],
            ),
        )


        limit = min(
            top_k,
            len(
                ranked
            ),
        )


        results = []


        for (
            index,
            score,
        ) in ranked[
            :limit
        ]:

            document = (
                self.documents[
                    index
                ]
            )


            results.append(
                self._result_from_document(
                    document=
                        document,

                    score=
                        float(
                            score
                        ),
                )
            )


        return results


    # ========================================================
    # Source-targeted BM25 candidate generation
    #
    # Used ONLY during structurally approved retry.
    #
    # Scores are still normal BM25 scores.
    # Source filtering only changes candidate eligibility.
    # ========================================================

    def search_by_sources(
        self,
        query: str,
        sources: list[str],
        top_k_per_source: int = 20,
    ) -> list[dict]:

        if (
            not query
            or
            not sources
            or
            top_k_per_source
            <=
            0
        ):

            return []


        query_tokens = (
            self._tokenize(
                query
            )
        )


        if not query_tokens:

            return []


        scores = (
            self.bm25.get_scores(
                query_tokens
            )
        )


        output = []

        seen_chunk_ids = set()


        for requested_source in sources:

            requested_source = (
                requested_source
                or ""
            ).strip()


            if not requested_source:

                continue


            matching = []


            for (
                index,
                score,
            ) in enumerate(
                scores
            ):

                document = (
                    self.documents[
                        index
                    ]
                )


                metadata = (
                    document.get(
                        "metadata",
                        {},
                    )
                    or {}
                )


                actual_source = (
                    metadata.get(
                        "source",
                        "",
                    )
                    or ""
                )


                if not (
                    self._source_matches(
                        actual_source=
                            actual_source,

                        requested_source=
                            requested_source,
                    )
                ):

                    continue


                matching.append(
                    (
                        index,
                        float(
                            score
                        ),
                    )
                )


            matching.sort(
                key=lambda item: (
                    -item[
                        1
                    ],
                    item[
                        0
                    ],
                )
            )


            for (
                index,
                score,
            ) in matching[
                :top_k_per_source
            ]:

                document = (
                    self.documents[
                        index
                    ]
                )


                chunk_id = str(
                    document[
                        "id"
                    ]
                )


                if (
                    chunk_id
                    in
                    seen_chunk_ids
                ):

                    continue


                seen_chunk_ids.add(
                    chunk_id
                )


                result = (
                    self._result_from_document(
                        document=
                            document,

                        score=
                            score,
                    )
                )


                # Diagnostic/provenance fields only.

                result[
                    "source_targeted"
                ] = True


                result[
                    "source_target"
                ] = (
                    requested_source
                )


                output.append(
                    result
                )


        return output