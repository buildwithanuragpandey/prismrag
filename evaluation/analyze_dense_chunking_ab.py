import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


RESULT_PATH = Path(
    "evaluation/results/dense_chunking_ab.json"
)


K_VALUES = (
    5,
    10,
    20,
)


def avg(
    values,
):

    values = list(
        values
    )

    if not values:
        return 0.0

    return mean(
        values
    )


def main():

    with open(
        RESULT_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )


    v1_results = {
        item["id"]: item
        for item
        in data["v1"]["per_example"]
    }


    v2_results = {
        item["id"]: item
        for item
        in data["v2"]["per_example"]
    }


    common_ids = sorted(
        set(
            v1_results
        )
        &
        set(
            v2_results
        )
    )


    print(
        "=" * 100
    )

    print(
        "DENSE CHUNKING A/B — PAIRED ANALYSIS"
    )

    print(
        "=" * 100
    )

    print(
        "Examples:",
        len(
            common_ids
        )
    )


    # ========================================================
    # Overall paired analysis
    # ========================================================

    for k in K_VALUES:

        key = str(
            k
        )


        recall_deltas = []

        ndcg_deltas = []

        mrr_deltas = []


        recall_wins = 0

        recall_losses = 0

        recall_ties = 0


        complete_wins = 0

        complete_losses = 0

        complete_ties = 0


        for example_id in common_ids:

            old = (
                v1_results[
                    example_id
                ][
                    "metrics"
                ][
                    key
                ]
            )

            new = (
                v2_results[
                    example_id
                ][
                    "metrics"
                ][
                    key
                ]
            )


            recall_delta = (
                new["recall"]
                -
                old["recall"]
            )


            recall_deltas.append(
                recall_delta
            )


            ndcg_deltas.append(
                new["ndcg"]
                -
                old["ndcg"]
            )


            mrr_deltas.append(
                new["mrr"]
                -
                old["mrr"]
            )


            if recall_delta > 0:

                recall_wins += 1

            elif recall_delta < 0:

                recall_losses += 1

            else:

                recall_ties += 1


            complete_delta = (
                new["complete"]
                -
                old["complete"]
            )


            if complete_delta > 0:

                complete_wins += 1

            elif complete_delta < 0:

                complete_losses += 1

            else:

                complete_ties += 1


        print(
            "\n"
            +
            "-" * 100
        )

        print(
            f"@{k}"
        )


        print(
            "Mean Recall delta:",
            round(
                avg(
                    recall_deltas
                ),
                4,
            )
        )

        print(
            "Mean MRR delta:",
            round(
                avg(
                    mrr_deltas
                ),
                4,
            )
        )

        print(
            "Mean nDCG delta:",
            round(
                avg(
                    ndcg_deltas
                ),
                4,
            )
        )


        print(
            "\nRecall paired:"
        )

        print(
            " Wins:",
            recall_wins
        )

        print(
            " Losses:",
            recall_losses
        )

        print(
            " Ties:",
            recall_ties
        )


        print(
            "\nComplete paired:"
        )

        print(
            " Wins:",
            complete_wins
        )

        print(
            " Losses:",
            complete_losses
        )

        print(
            " Ties:",
            complete_ties
        )


    # ========================================================
    # By question type
    # ========================================================

    grouped = defaultdict(
        list
    )


    for example_id in common_ids:

        question_type = (
            v1_results[
                example_id
            ].get(
                "question_type",
                "unknown",
            )
        )


        grouped[
            question_type
        ].append(
            example_id
        )


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "BY QUESTION TYPE"
    )

    print(
        "=" * 100
    )


    for question_type, ids in sorted(
        grouped.items()
    ):

        print(
            "\n"
            +
            question_type
        )

        print(
            "Count:",
            len(
                ids
            )
        )


        for k in K_VALUES:

            key = str(
                k
            )


            recall_deltas = []

            complete_deltas = []

            mrr_deltas = []

            ndcg_deltas = []


            for example_id in ids:

                old = (
                    v1_results[
                        example_id
                    ][
                        "metrics"
                    ][
                        key
                    ]
                )

                new = (
                    v2_results[
                        example_id
                    ][
                        "metrics"
                    ][
                        key
                    ]
                )


                recall_deltas.append(
                    new["recall"]
                    -
                    old["recall"]
                )


                complete_deltas.append(
                    new["complete"]
                    -
                    old["complete"]
                )


                mrr_deltas.append(
                    new["mrr"]
                    -
                    old["mrr"]
                )


                ndcg_deltas.append(
                    new["ndcg"]
                    -
                    old["ndcg"]
                )


            print(
                f" @{k} "
                f"Recall={avg(recall_deltas):+.4f} "
                f"Complete={avg(complete_deltas):+.4f} "
                f"MRR={avg(mrr_deltas):+.4f} "
                f"nDCG={avg(ndcg_deltas):+.4f}"
            )


    # ========================================================
    # Largest Recall improvements / regressions @10
    # ========================================================

    comparisons = []


    for example_id in common_ids:

        old = (
            v1_results[
                example_id
            ][
                "metrics"
            ][
                "10"
            ]
        )

        new = (
            v2_results[
                example_id
            ][
                "metrics"
            ][
                "10"
            ]
        )


        comparisons.append(
            {
                "id":
                    example_id,

                "question_type":
                    v1_results[
                        example_id
                    ].get(
                        "question_type"
                    ),

                "recall_delta":
                    new["recall"]
                    -
                    old["recall"],

                "complete_delta":
                    new["complete"]
                    -
                    old["complete"],

                "v1_recall":
                    old["recall"],

                "v2_recall":
                    new["recall"],

                "v1_documents":
                    v1_results[
                        example_id
                    ][
                        "retrieved_document_ids"
                    ][
                        :10
                    ],

                "v2_documents":
                    v2_results[
                        example_id
                    ][
                        "retrieved_document_ids"
                    ][
                        :10
                    ],

                "gold":
                    v1_results[
                        example_id
                    ][
                        "gold_document_ids"
                    ],
            }
        )


    comparisons.sort(
        key=lambda item: (
            item[
                "recall_delta"
            ]
        ),
        reverse=True,
    )


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "TOP RECALL IMPROVEMENTS @10"
    )

    print(
        "=" * 100
    )


    improvements = [
        item
        for item
        in comparisons
        if item[
            "recall_delta"
        ] > 0
    ]


    for item in improvements[:10]:

        print(
            "\n",
            item[
                "id"
            ],
            item[
                "question_type"
            ],
        )

        print(
            "Recall:",
            item[
                "v1_recall"
            ],
            "->",
            item[
                "v2_recall"
            ],
        )

        print(
            "Gold:",
            item[
                "gold"
            ]
        )

        print(
            "V1:",
            item[
                "v1_documents"
            ]
        )

        print(
            "V2:",
            item[
                "v2_documents"
            ]
        )


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "TOP RECALL REGRESSIONS @10"
    )

    print(
        "=" * 100
    )


    regressions = sorted(

        (
            item
            for item
            in comparisons
            if item[
                "recall_delta"
            ] < 0
        ),

        key=lambda item: (
            item[
                "recall_delta"
            ]
        ),
    )


    for item in regressions[:10]:

        print(
            "\n",
            item[
                "id"
            ],
            item[
                "question_type"
            ],
        )

        print(
            "Recall:",
            item[
                "v1_recall"
            ],
            "->",
            item[
                "v2_recall"
            ],
        )

        print(
            "Gold:",
            item[
                "gold"
            ]
        )

        print(
            "V1:",
            item[
                "v1_documents"
            ]
        )

        print(
            "V2:",
            item[
                "v2_documents"
            ]
        )


if __name__ == "__main__":

    main()