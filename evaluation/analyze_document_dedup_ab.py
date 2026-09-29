import json
import math
from pathlib import Path
from statistics import mean


RESULT_PATH = Path(
    "evaluation/results/dense_chunking_ab.json"
)


K_VALUES = (
    1,
    3,
    5,
    10,
    20,
)


def unique_documents(
    document_ids,
):

    seen = set()

    output = []


    for document_id in document_ids:

        if not document_id:
            continue

        if document_id in seen:
            continue

        seen.add(
            document_id
        )

        output.append(
            document_id
        )


    return output


def evaluate_document_ranking(
    document_ids,
    gold_document_ids,
    k,
):

    ranked = (
        unique_documents(
            document_ids
        )[
            :k
        ]
    )


    gold = set(
        gold_document_ids
    )


    retrieved = set(
        ranked
    )


    relevant = (
        gold
        &
        retrieved
    )


    recall = (
        len(
            relevant
        )
        /
        len(
            gold
        )
        if gold
        else 0.0
    )


    hit = float(
        bool(
            relevant
        )
    )


    complete = float(
        bool(
            gold
        )
        and
        gold.issubset(
            retrieved
        )
    )


    # ========================================================
    # MRR
    # ========================================================

    mrr = 0.0


    for rank, document_id in enumerate(
        ranked,
        start=1,
    ):

        if document_id in gold:

            mrr = (
                1.0
                /
                rank
            )

            break


    # ========================================================
    # nDCG
    # ========================================================

    dcg = 0.0


    for rank, document_id in enumerate(
        ranked,
        start=1,
    ):

        if document_id in gold:

            dcg += (
                1.0
                /
                math.log2(
                    rank
                    +
                    1
                )
            )


    ideal_count = min(
        len(
            gold
        ),
        k,
    )


    idcg = sum(

        1.0
        /
        math.log2(
            rank
            +
            1
        )

        for rank
        in range(
            1,
            ideal_count
            +
            1
        )
    )


    ndcg = (
        dcg
        /
        idcg
        if idcg
        else 0.0
    )


    return {
        "recall":
            recall,

        "hit":
            hit,

        "complete":
            complete,

        "mrr":
            mrr,

        "ndcg":
            ndcg,
    }


def analyze_system(
    per_example,
):

    summary = {}


    for k in K_VALUES:

        metrics = {
            "recall": [],
            "hit": [],
            "complete": [],
            "mrr": [],
            "ndcg": [],
        }


        for example in per_example:

            values = (
                evaluate_document_ranking(

                    example[
                        "retrieved_document_ids"
                    ],

                    example[
                        "gold_document_ids"
                    ],

                    k,
                )
            )


            for name, value in (
                values.items()
            ):

                metrics[
                    name
                ].append(
                    value
                )


        summary[
            str(
                k
            )
        ] = {

            name:
                mean(
                    values
                )

            for name, values
            in metrics.items()
        }


    return summary


def paired_analysis(
    v1_examples,
    v2_examples,
):

    v1_map = {
        item["id"]:
            item
        for item
        in v1_examples
    }


    v2_map = {
        item["id"]:
            item
        for item
        in v2_examples
    }


    common_ids = sorted(
        set(
            v1_map
        )
        &
        set(
            v2_map
        )
    )


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "PAIRED DOCUMENT-DEDUP ANALYSIS"
    )

    print(
        "=" * 100
    )


    for k in (
        5,
        10,
        20,
    ):

        wins = 0
        losses = 0
        ties = 0

        complete_wins = 0
        complete_losses = 0
        complete_ties = 0


        for example_id in common_ids:

            old = (
                evaluate_document_ranking(

                    v1_map[
                        example_id
                    ][
                        "retrieved_document_ids"
                    ],

                    v1_map[
                        example_id
                    ][
                        "gold_document_ids"
                    ],

                    k,
                )
            )


            new = (
                evaluate_document_ranking(

                    v2_map[
                        example_id
                    ][
                        "retrieved_document_ids"
                    ],

                    v2_map[
                        example_id
                    ][
                        "gold_document_ids"
                    ],

                    k,
                )
            )


            recall_delta = (
                new["recall"]
                -
                old["recall"]
            )


            if recall_delta > 0:
                wins += 1

            elif recall_delta < 0:
                losses += 1

            else:
                ties += 1


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
            f"\n@{k}"
        )

        print(
            "Recall wins:",
            wins
        )

        print(
            "Recall losses:",
            losses
        )

        print(
            "Recall ties:",
            ties
        )

        print(
            "Complete wins:",
            complete_wins
        )

        print(
            "Complete losses:",
            complete_losses
        )

        print(
            "Complete ties:",
            complete_ties
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


    v1_examples = (
        data[
            "v1"
        ][
            "per_example"
        ]
    )


    v2_examples = (
        data[
            "v2"
        ][
            "per_example"
        ]
    )


    v1_summary = (
        analyze_system(
            v1_examples
        )
    )


    v2_summary = (
        analyze_system(
            v2_examples
        )
    )


    print(
        "=" * 100
    )

    print(
        "DOCUMENT-DEDUP DENSE A/B"
    )

    print(
        "=" * 100
    )


    for k in K_VALUES:

        key = str(
            k
        )


        print(
            f"\n@{k}"
        )


        for metric in (
            "recall",
            "hit",
            "complete",
            "mrr",
            "ndcg",
        ):

            old = (
                v1_summary[
                    key
                ][
                    metric
                ]
            )

            new = (
                v2_summary[
                    key
                ][
                    metric
                ]
            )


            delta = (
                new
                -
                old
            )


            print(
                f"{metric:10s} "
                f"V1={old:.4f} "
                f"V2={new:.4f} "
                f"delta={delta:+.4f}"
            )


    paired_analysis(
        v1_examples,
        v2_examples,
    )


if __name__ == "__main__":

    main()