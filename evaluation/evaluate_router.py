import argparse
import json

from collections import Counter, defaultdict
from pathlib import Path

from adaptive_agentic_rag.agents.query_router import (
    QueryRouter
)


DEFAULT_DATASET_PATH = Path(
    "evaluation/datasets/router_dev_control_180.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "evaluation/results/router_dev.json"
)


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH
    )


    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH
    )


    return parser.parse_args()


# ============================================================
# Load
# ============================================================

def load_examples(
    path: Path
):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ============================================================
# Evaluation
# ============================================================

def main():

    args = parse_args()


    examples = load_examples(
        args.dataset
    )


    router = QueryRouter()


    predictions = []


    query_type_correct = 0
    strategy_correct = 0
    rerank_correct = 0
    mmr_correct = 0
    exact_correct = 0


    confusion = defaultdict(
        Counter
    )


    route_distribution = Counter()


    #
    # Strategy-level diagnostics
    #

    gold_simple_count = 0

    predicted_dense_count = 0

    correct_simple_dense = 0

    hard_count = 0

    hard_sent_to_dense = 0


    for example in examples:

        decision = router.route(
            example[
                "question"
            ]
        )


        predicted_type = decision[
            "query_type"
        ]

        predicted_strategy = decision[
            "retrieval_strategy"
        ]

        predicted_rerank = decision[
            "rerank"
        ]

        predicted_mmr = decision[
            "mmr"
        ]


        gold_type = example[
            "gold_query_type"
        ]

        gold_strategy = example[
            "gold_retrieval_strategy"
        ]

        gold_rerank = example[
            "gold_rerank"
        ]

        gold_mmr = example[
            "gold_mmr"
        ]


        type_ok = (
            predicted_type
            ==
            gold_type
        )


        strategy_ok = (
            predicted_strategy
            ==
            gold_strategy
        )


        rerank_ok = (
            predicted_rerank
            ==
            gold_rerank
        )


        mmr_ok = (
            predicted_mmr
            ==
            gold_mmr
        )


        exact_ok = (
            type_ok
            and strategy_ok
            and rerank_ok
            and mmr_ok
        )


        query_type_correct += int(
            type_ok
        )

        strategy_correct += int(
            strategy_ok
        )

        rerank_correct += int(
            rerank_ok
        )

        mmr_correct += int(
            mmr_ok
        )

        exact_correct += int(
            exact_ok
        )


        confusion[
            gold_type
        ][
            predicted_type
        ] += 1


        route_key = (
            f"{predicted_type}"
            f"|{predicted_strategy}"
            f"|rerank={predicted_rerank}"
            f"|mmr={predicted_mmr}"
        )


        route_distribution[
            route_key
        ] += 1


        # ----------------------------------------------------
        # Simple-route diagnostics
        # ----------------------------------------------------

        if gold_type == "simple":

            gold_simple_count += 1


            if (
                predicted_strategy
                ==
                "dense"
            ):

                correct_simple_dense += 1


        else:

            hard_count += 1


            if (
                predicted_strategy
                ==
                "dense"
            ):

                hard_sent_to_dense += 1


        if (
            predicted_strategy
            ==
            "dense"
        ):

            predicted_dense_count += 1


        predictions.append(
            {
                "id":
                    example[
                        "id"
                    ],

                "question":
                    example[
                        "question"
                    ],

                "source":
                    example[
                        "source"
                    ],

                "gold":
                    {
                        "query_type":
                            gold_type,

                        "retrieval_strategy":
                            gold_strategy,

                        "rerank":
                            gold_rerank,

                        "mmr":
                            gold_mmr
                    },

                "predicted":
                    decision,

                "correct":
                    exact_ok
            }
        )


    total = len(
        examples
    )


    simple_route_recall = (
        correct_simple_dense
        /
        gold_simple_count

        if gold_simple_count
        else 0.0
    )


    simple_route_precision = (
        correct_simple_dense
        /
        predicted_dense_count

        if predicted_dense_count
        else 0.0
    )


    hard_to_dense_error_rate = (
        hard_sent_to_dense
        /
        hard_count

        if hard_count
        else 0.0
    )


    summary = {

        "total":
            total,

        "query_type_accuracy":
            (
                query_type_correct
                /
                total
            ),

        "strategy_accuracy":
            (
                strategy_correct
                /
                total
            ),

        "rerank_accuracy":
            (
                rerank_correct
                /
                total
            ),

        "mmr_accuracy":
            (
                mmr_correct
                /
                total
            ),

        "exact_decision_accuracy":
            (
                exact_correct
                /
                total
            ),

        "simple_route_precision":
            simple_route_precision,

        "simple_route_recall":
            simple_route_recall,

        "hard_to_dense_error_rate":
            hard_to_dense_error_rate,

        "route_distribution":
            dict(
                route_distribution
            ),

        "confusion_matrix":
            {
                gold:
                    dict(row)

                for gold, row
                in confusion.items()
            }
    }


    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "dataset":
                    str(
                        args.dataset
                    ),

                "summary":
                    summary,

                "predictions":
                    predictions
            },
            file,
            ensure_ascii=False,
            indent=2
        )


    print(
        "\n===== ROUTER EVALUATION ====="
    )


    print(
        "Dataset:",
        args.dataset
    )


    print(
        "Total:",
        total
    )


    print(
        "Query-type accuracy:",
        round(
            summary[
                "query_type_accuracy"
            ],
            4
        )
    )


    print(
        "Strategy accuracy:",
        round(
            summary[
                "strategy_accuracy"
            ],
            4
        )
    )


    print(
        "Exact decision accuracy:",
        round(
            summary[
                "exact_decision_accuracy"
            ],
            4
        )
    )


    print(
        "\n===== ROUTING SAFETY ====="
    )


    print(
        "Simple-route precision:",
        round(
            summary[
                "simple_route_precision"
            ],
            4
        )
    )


    print(
        "Simple-route recall:",
        round(
            summary[
                "simple_route_recall"
            ],
            4
        )
    )


    print(
        "Hard → Dense error rate:",
        round(
            summary[
                "hard_to_dense_error_rate"
            ],
            4
        )
    )


    print(
        "\n===== CONFUSION MATRIX ====="
    )


    for gold_type, row in (
        summary[
            "confusion_matrix"
        ].items()
    ):

        print(
            gold_type,
            "→",
            row
        )


    print(
        "\n===== ROUTE DISTRIBUTION ====="
    )


    for route, count in (
        summary[
            "route_distribution"
        ].items()
    ):

        print(
            count,
            route
        )


    print(
        "\nSaved:",
        args.output
    )


if __name__ == "__main__":
    main()