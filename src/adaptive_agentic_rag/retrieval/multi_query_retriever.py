from adaptive_agentic_rag.retrieval.query_decomposer import (
    QueryDecomposer
)


class MultiQueryRetriever:
    """
    Candidate generator for hard multi-hop queries.

    Important:

    This class performs only Hybrid retrieval.

    Cross-encoder reranking is intentionally performed
    later, ONCE, using the original user query.
    """

    def __init__(
        self,
        hybrid_retriever,
        decomposer: QueryDecomposer | None = None,
        per_query_top_k: int = 20,
        rrf_k: int = 60,
        original_query_weight: float = 1.5
    ):

        self.hybrid = (
            hybrid_retriever
        )


        self.decomposer = (
            decomposer
            or
            QueryDecomposer()
        )


        self.per_query_top_k = (
            per_query_top_k
        )


        self.rrf_k = (
            rrf_k
        )


        self.original_query_weight = (
            original_query_weight
        )


    # ========================================================
    # Stable candidate key
    # ========================================================

    def _candidate_key(
        self,
        item: dict
    ) -> str:

        chunk_id = item.get(
            "id"
        )


        if chunk_id:

            return str(
                chunk_id
            )


        document_id = item.get(
            "document_id",
            ""
        )


        text = item.get(
            "text",
            ""
        )


        return (
            f"{document_id}"
            f"::{text[:200]}"
        )


    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 40
    ) -> list[dict]:

        queries = (
            self.decomposer.decompose(
                query
            )
        )


        if not queries:

            return []


        fused = {}


        # ====================================================
        # Hybrid retrieval for every facet
        # ====================================================

        for query_index, subquery in enumerate(
            queries
        ):

            weight = (

                self.original_query_weight

                if query_index == 0

                else 1.0
            )


            results = (
                self.hybrid.search(

                    subquery,

                    top_k=(
                        self.per_query_top_k
                    )
                )
            )


            for rank, item in enumerate(
                results,
                start=1
            ):

                key = (
                    self._candidate_key(
                        item
                    )
                )


                rrf_score = (

                    weight

                    /

                    (
                        self.rrf_k
                        +
                        rank
                    )
                )


                if key not in fused:

                    fused[
                        key
                    ] = {

                        "item":
                            dict(
                                item
                            ),

                        "fusion_score":
                            0.0,

                        "matched_queries":
                            [],

                        "best_rank":
                            rank
                    }


                entry = fused[
                    key
                ]


                entry[
                    "fusion_score"
                ] += rrf_score


                entry[
                    "best_rank"
                ] = min(

                    entry[
                        "best_rank"
                    ],

                    rank
                )


                if (
                    subquery
                    not in
                    entry[
                        "matched_queries"
                    ]
                ):

                    entry[
                        "matched_queries"
                    ].append(
                        subquery
                    )


                #
                # Prefer a candidate copy that already
                # contains a vector.
                #

                if (
                    "vector"
                    not in
                    entry[
                        "item"
                    ]
                    and
                    "vector"
                    in item
                ):

                    entry[
                        "item"
                    ][
                        "vector"
                    ] = item[
                        "vector"
                    ]


        # ====================================================
        # Convert fused map to candidates
        # ====================================================

        ranked = []


        for entry in fused.values():

            item = dict(
                entry[
                    "item"
                ]
            )


            item[
                "multi_query_score"
            ] = entry[
                "fusion_score"
            ]


            item[
                "multi_query_best_rank"
            ] = entry[
                "best_rank"
            ]


            item[
                "matched_queries"
            ] = entry[
                "matched_queries"
            ]


            ranked.append(
                item
            )


        ranked.sort(

            key=lambda item: (

                -item[
                    "multi_query_score"
                ],

                item[
                    "multi_query_best_rank"
                ]
            )
        )


        return ranked[
            :top_k
        ]