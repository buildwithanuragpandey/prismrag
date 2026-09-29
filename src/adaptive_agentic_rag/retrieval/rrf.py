from collections import defaultdict


def reciprocal_rank_fusion(
    result_lists,
    top_k: int = 20,
    k: int = 60,
):

    if top_k <= 0:

        return []


    scores = defaultdict(
        float
    )


    documents = {}


    # ========================================================
    # Fuse rankings
    # ========================================================

    for results in result_lists:

        for rank, document in enumerate(
            results,
            start=1,
        ):

            document_id = (
                document[
                    "id"
                ]
            )


            # =================================================
            # Standard Reciprocal Rank Fusion
            #
            # RRF deliberately ignores incompatible source
            # score scales such as:
            #
            # Dense cosine score
            # BM25 score
            #
            # and uses ranking positions instead.
            # =================================================

            scores[
                document_id
            ] += (
                1.0
                /
                (
                    k
                    +
                    rank
                )
            )


            # =================================================
            # Preserve the first document representation.
            # =================================================

            if (
                document_id
                not in documents
            ):

                documents[
                    document_id
                ] = (
                    document.copy()
                )


            else:

                # ---------------------------------------------
                # Dense results generally carry vectors,
                # BM25 results generally do not.
                #
                # Preserve the vector if it appears in another
                # retrieval list.
                # ---------------------------------------------

                if (
                    "vector"
                    in document
                    and
                    "vector"
                    not in documents[
                        document_id
                    ]
                ):

                    documents[
                        document_id
                    ][
                        "vector"
                    ] = (
                        document[
                            "vector"
                        ]
                    )


    # ========================================================
    # Rank fused candidates
    # ========================================================

    ranked = sorted(

        scores.items(),

        key=lambda item:
            item[1],

        reverse=True,
    )


    output = []


    for (
        document_id,
        rrf_score,
    ) in ranked[
        :top_k
    ]:

        item = (
            documents[
                document_id
            ].copy()
        )


        # ====================================================
        # Canonical downstream score at the Hybrid stage
        # ====================================================

        item[
            "score"
        ] = float(
            rrf_score
        )


        # ====================================================
        # Preserve explicit provenance / observability
        #
        # Later the Cross-Encoder may replace item["score"],
        # but this diagnostic value survives.
        # ====================================================

        item[
            "rrf_score"
        ] = float(
            rrf_score
        )


        output.append(
            item
        )


    return output