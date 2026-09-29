import json
import math
import statistics
import time
from pathlib import Path

from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever,
)


EVAL_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)

OUTPUT_PATH = Path(
    "evaluation/results/"
    "dense_representation_ab.json"
)


V2A_COLLECTION = (
    "multihop_chunks_v2"
)

V2B_COLLECTION = (
    "multihop_chunks_v2b"
)


K_VALUES = (
    1,
    3,
    5,
    10,
    20,
)


# Retrieve deeper than the evaluation depth so that
# document-level dedup does not run out of candidates.
CANDIDATE_K = 50


def unique_preserve_order(
    values,
):

    seen = set()
    output = []

    for value in values:

        if not value:
            continue

        if value in seen:
            continue

        seen.add(value)
        output.append(value)

    return output


def evaluate_ranking(
    document_ids,
    gold_document_ids,
    k,
    deduplicate=False,
):

    if deduplicate:

        ranked = (
            unique_preserve_order(
                document_ids
            )[:k]
        )

    else:

        ranked = (
            document_ids[:k]
        )


    gold = set(
        gold_document_ids
    )


    unique_retrieved = set(
        ranked
    )


    relevant = (
        gold
        &
        unique_retrieved
    )


    recall = (
        len(relevant)
        /
        len(gold)
        if gold
        else 0.0
    )


    hit = float(
        bool(relevant)
    )


    complete = float(
        bool(gold)
        and
        gold.issubset(
            unique_retrieved
        )
    )


    # ========================================================
    # MRR
    # ========================================================

    mrr = 0.0


    seen = set()


    for rank, document_id in enumerate(
        ranked,
        start=1,
    ):

        # For raw chunk ranking, duplicate chunks from the
        # same document should not generate another relevance
        # event.
        if document_id in seen:

            continue

        seen.add(
            document_id
        )


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

    seen = set()


    for rank, document_id in enumerate(
        ranked,
        start=1,
    ):

        if (
            document_id in gold
            and
            document_id not in seen
        ):

            dcg += (
                1.0
                /
                math.log2(
                    rank + 1
                )
            )


        seen.add(
            document_id
        )


    ideal_count = min(
        len(gold),
        k,
    )


    idcg = sum(

        1.0
        /
        math.log2(
            rank + 1
        )

        for rank in range(
            1,
            ideal_count + 1,
        )
    )


    ndcg = (
        dcg / idcg
        if idcg
        else 0.0
    )


    return {
        "recall": recall,
        "hit": hit,
        "complete": complete,
        "mrr": mrr,
        "ndcg": ndcg,
    }


def percentile(
    values,
    p,
):

    ordered = sorted(values)

    position = (
        len(ordered) - 1
    ) * p

    lower = math.floor(
        position
    )

    upper = math.ceil(
        position
    )


    if lower == upper:

        return float(
            ordered[lower]
        )


    fraction = (
        position - lower
    )


    return float(

        ordered[lower]
        *
        (1 - fraction)

        +

        ordered[upper]
        *
        fraction
    )


def benchmark(
    examples,
    collection_name,
    label,
):

    print(
        "\n"
        +
        "=" * 100
    )

    print(label)

    print(
        "Collection:",
        collection_name,
    )

    print(
        "=" * 100
    )


    retriever = DenseRetriever(
        collection_name=
            collection_name
    )


    per_example = []

    latencies = []


    try:

        for index, example in enumerate(
            examples,
            start=1,
        ):

            gold_document_ids = (
                unique_preserve_order(

                    evidence[
                        "document_id"
                    ]

                    for evidence
                    in example.get(
                        "evidence",
                        []
                    )
                )
            )


            started = (
                time.perf_counter()
            )


            results = (
                retriever.search(

                    example[
                        "question"
                    ],

                    top_k=
                        CANDIDATE_K,
                )
            )


            latency_ms = (
                time.perf_counter()
                -
                started
            ) * 1000


            latencies.append(
                latency_ms
            )


            document_ids = [

                item.get(
                    "document_id"
                )

                for item
                in results
            ]


            metrics = {
                "chunk_budget": {},
                "document_budget": {},
            }


            for k in K_VALUES:

                metrics[
                    "chunk_budget"
                ][
                    str(k)
                ] = (
                    evaluate_ranking(
                        document_ids,
                        gold_document_ids,
                        k,
                        deduplicate=False,
                    )
                )


                metrics[
                    "document_budget"
                ][
                    str(k)
                ] = (
                    evaluate_ranking(
                        document_ids,
                        gold_document_ids,
                        k,
                        deduplicate=True,
                    )
                )


            per_example.append(
                {
                    "id":
                        example["id"],

                    "question_type":
                        example.get(
                            "question_type"
                        ),

                    "gold_document_ids":
                        gold_document_ids,

                    "retrieved_document_ids":
                        document_ids,

                    "retrieved_chunk_ids":
                        [
                            item.get("id")
                            for item
                            in results
                        ],

                    "latency_ms":
                        latency_ms,

                    "metrics":
                        metrics,
                }
            )


            if (
                index % 50 == 0
                or
                index == len(examples)
            ):

                print(
                    f"{index}/{len(examples)}"
                )


    finally:

        retriever.close()


    summary = {
        "chunk_budget": {},
        "document_budget": {},
    }


    for mode in (
        "chunk_budget",
        "document_budget",
    ):

        for k in K_VALUES:

            key = str(k)


            summary[
                mode
            ][
                key
            ] = {}


            for metric in (
                "recall",
                "hit",
                "complete",
                "mrr",
                "ndcg",
            ):

                values = [

                    item[
                        "metrics"
                    ][
                        mode
                    ][
                        key
                    ][
                        metric
                    ]

                    for item
                    in per_example
                ]


                summary[
                    mode
                ][
                    key
                ][
                    metric
                ] = (
                    statistics.mean(
                        values
                    )
                )


    latency = {
        "mean_ms":
            statistics.mean(
                latencies
            ),

        "p50_ms":
            percentile(
                latencies,
                0.50,
            ),

        "p95_ms":
            percentile(
                latencies,
                0.95,
            ),
    }


    return {
        "label": label,
        "collection": collection_name,
        "summary": summary,
        "latency": latency,
        "per_example": per_example,
    }


def print_comparison(
    a,
    b,
):

    print(
        "\n"
        +
        "=" * 110
    )

    print(
        "DENSE REPRESENTATION A/B"
    )

    print(
        "=" * 110
    )


    for mode in (
        "chunk_budget",
        "document_budget",
    ):

        print(
            "\n"
            +
            mode.upper()
        )


        for k in K_VALUES:

            key = str(k)

            print(
                f"\n@{k}"
            )


            for metric in (
                "recall",
                "complete",
                "mrr",
                "ndcg",
            ):

                old = (
                    a[
                        "summary"
                    ][
                        mode
                    ][
                        key
                    ][
                        metric
                    ]
                )

                new = (
                    b[
                        "summary"
                    ][
                        mode
                    ][
                        key
                    ][
                        metric
                    ]
                )


                print(
                    f"{metric:10s} "
                    f"V2-A={old:.4f} "
                    f"V2-B={new:.4f} "
                    f"delta={new-old:+.4f}"
                )


    print(
        "\nLATENCY"
    )


    for metric in (
        "mean_ms",
        "p50_ms",
        "p95_ms",
    ):

        old = (
            a[
                "latency"
            ][
                metric
            ]
        )

        new = (
            b[
                "latency"
            ][
                metric
            ]
        )


        print(
            f"{metric:10s} "
            f"V2-A={old:.2f} "
            f"V2-B={new:.2f} "
            f"delta={new-old:+.2f}"
        )


def main():

    print(
        "Loading frozen evaluation set..."
    )


    with open(
        EVAL_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        examples = json.load(
            file
        )


    answerable = [

        example

        for example
        in examples

        if example.get(
            "evidence"
        )
    ]


    print(
        "Answerable examples:",
        len(answerable),
    )


    v2a = benchmark(
        answerable,
        V2A_COLLECTION,
        "V2-A — chunk text",
    )


    v2b = benchmark(
        answerable,
        V2B_COLLECTION,
        "V2-B — title + source + chunk text",
    )


    print_comparison(
        v2a,
        v2b,
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "experiment":
                    "dense_representation_ab",

                "candidate_k":
                    CANDIDATE_K,

                "v2a":
                    v2a,

                "v2b":
                    v2b,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


    print(
        "\nSaved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()