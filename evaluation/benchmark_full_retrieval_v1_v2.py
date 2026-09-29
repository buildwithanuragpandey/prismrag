import gc
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever,
)

from adaptive_agentic_rag.retrieval.reranked_retriever import (
    RerankedRetriever,
)


# ============================================================
# Configuration
# ============================================================

EVAL_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)

OUTPUT_PATH = Path(
    "evaluation/results/"
    "full_retrieval_v1_v2.json"
)


SYSTEMS = {
    "v1": {
        "label":
            "V1 — 2000-char chunks",

        "collection":
            "multihop_chunks",

        "corpus":
            (
                "data/processed/"
                "processed_corpus.json"
            ),
    },

    "v2": {
        "label":
            "V2-A — 1000-char chunks",

        "collection":
            "multihop_chunks_v2",

        "corpus":
            (
                "data/processed/"
                "processed_corpus_v2.json"
            ),
    },
}


K_VALUES = (
    1,
    3,
    5,
    10,
    20,
)


FINAL_TOP_K = max(
    K_VALUES
)


# ============================================================
# Helpers
# ============================================================

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

        seen.add(
            value
        )

        output.append(
            value
        )

    return output


def percentile(
    values,
    p,
):

    if not values:
        return 0.0

    ordered = sorted(
        values
    )

    position = (
        len(ordered) - 1
    ) * p

    lower = int(
        math.floor(
            position
        )
    )

    upper = int(
        math.ceil(
            position
        )
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
        (1.0 - fraction)

        +

        ordered[upper]
        *
        fraction
    )


# ============================================================
# Retrieval metrics
#
# These remain DATASET EVIDENCE metrics.
# We already know MultiHopRAG contains some noisy evidence
# annotations, so do not interpret these as perfect semantic
# ground truth.
# ============================================================

def evaluate_at_k(
    results,
    gold_document_ids,
    k,
):

    selected = (
        results[:k]
    )

    document_ids = [

        item.get(
            "document_id"
        )

        for item
        in selected

        if item.get(
            "document_id"
        )
    ]


    unique_documents = (
        unique_preserve_order(
            document_ids
        )
    )


    gold = set(
        gold_document_ids
    )


    retrieved = set(
        unique_documents
    )


    relevant = (
        gold
        &
        retrieved
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
            retrieved
        )
    )


    # ========================================================
    # MRR
    # ========================================================

    mrr = 0.0


    for rank, document_id in enumerate(
        document_ids,
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
    #
    # Duplicate chunks from one document do not receive
    # relevance credit multiple times.
    # ========================================================

    dcg = 0.0

    seen = set()


    for rank, document_id in enumerate(
        document_ids,
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


    result_count = len(
        document_ids
    )

    unique_count = len(
        unique_documents
    )


    duplicate_rate = (
        1.0
        -
        (
            unique_count
            /
            result_count
        )

        if result_count
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

        "unique_documents":
            unique_count,

        "duplicate_rate":
            duplicate_rate,
    }


# ============================================================
# Aggregate
# ============================================================

def aggregate_metrics(
    examples,
):

    output = {}


    for k in K_VALUES:

        key = str(k)

        metric_names = (
            "recall",
            "hit",
            "complete",
            "mrr",
            "ndcg",
            "unique_documents",
            "duplicate_rate",
        )


        output[key] = {}


        for metric_name in metric_names:

            values = [

                item[
                    "metrics"
                ][
                    key
                ][
                    metric_name
                ]

                for item
                in examples
            ]


            output[
                key
            ][
                metric_name
            ] = (
                statistics.mean(
                    values
                )
                if values
                else 0.0
            )


    return output


def aggregate_by_question_type(
    examples,
):

    grouped = defaultdict(
        list
    )


    for item in examples:

        grouped[
            item.get(
                "question_type",
                "unknown",
            )
        ].append(
            item
        )


    return {

        question_type:
            aggregate_metrics(
                items
            )

        for question_type, items
        in sorted(
            grouped.items()
        )
    }


# ============================================================
# Run one complete retrieval system
# ============================================================

def benchmark_system(
    examples,
    config,
):

    print(
        "\n"
        +
        "=" * 110
    )

    print(
        config[
            "label"
        ]
    )

    print(
        "Dense collection:",
        config[
            "collection"
        ]
    )

    print(
        "BM25 corpus:",
        config[
            "corpus"
        ]
    )

    print(
        "=" * 110
    )


    dense = DenseRetriever(
        collection_name=
            config[
                "collection"
            ]
    )


    retriever = RerankedRetriever(
        dense_retriever=dense,
        bm25_corpus_path=
            config[
                "corpus"
            ],
    )


    per_example = []

    latencies = []

    errors = []


    try:

        total = len(
            examples
        )


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


            error = None


            try:

                results = (
                    retriever.search(
                        example[
                            "question"
                        ],
                        top_k=FINAL_TOP_K,
                    )
                )

            except Exception as exc:

                results = []

                error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )


            latency_ms = (
                time.perf_counter()
                -
                started
            ) * 1000.0


            latencies.append(
                latency_ms
            )


            if error is not None:

                errors.append(
                    {
                        "id":
                            example["id"],

                        "error":
                            error,
                    }
                )


            metrics = {}


            for k in K_VALUES:

                metrics[
                    str(k)
                ] = (
                    evaluate_at_k(
                        results,
                        gold_document_ids,
                        k,
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
                        [
                            item.get(
                                "document_id"
                            )
                            for item
                            in results
                        ],

                    "retrieved_chunk_ids":
                        [
                            item.get(
                                "id"
                            )
                            for item
                            in results
                        ],

                    "latency_ms":
                        latency_ms,

                    "error":
                        error,

                    "metrics":
                        metrics,
                }
            )


            if (
                index % 25 == 0
                or
                index == total
            ):

                print(
                    f"{index}/{total}"
                )


    finally:

        retriever.close()


    return {
        "label":
            config[
                "label"
            ],

        "collection":
            config[
                "collection"
            ],

        "corpus":
            config[
                "corpus"
            ],

        "errors":
            errors,

        "error_count":
            len(
                errors
            ),

        "summary":
            aggregate_metrics(
                per_example
            ),

        "by_question_type":
            aggregate_by_question_type(
                per_example
            ),

        "latency":
            {
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
            },

        "per_example":
            per_example,
    }


# ============================================================
# Report
# ============================================================

def print_comparison(
    v1,
    v2,
):

    print(
        "\n"
        +
        "=" * 110
    )

    print(
        "FULL RETRIEVAL STACK — V1 vs V2-A"
    )

    print(
        "=" * 110
    )


    print(
        "\nExecution errors:"
    )

    print(
        "V1:",
        v1[
            "error_count"
        ]
    )

    print(
        "V2:",
        v2[
            "error_count"
        ]
    )


    for k in K_VALUES:

        key = str(k)

        print(
            f"\n@{k}"
        )


        for metric in (
            "recall",
            "hit",
            "complete",
            "mrr",
            "ndcg",
            "duplicate_rate",
        ):

            old = (
                v1[
                    "summary"
                ][
                    key
                ][
                    metric
                ]
            )

            new = (
                v2[
                    "summary"
                ][
                    key
                ][
                    metric
                ]
            )


            print(
                f"{metric:16s} "
                f"V1={old:.4f} "
                f"V2={new:.4f} "
                f"delta={new-old:+.4f}"
            )


    print(
        "\n"
        +
        "=" * 110
    )

    print(
        "BY QUESTION TYPE @10"
    )

    print(
        "=" * 110
    )


    all_types = sorted(
        set(
            v1[
                "by_question_type"
            ]
        )
        |
        set(
            v2[
                "by_question_type"
            ]
        )
    )


    for question_type in all_types:

        print(
            "\n",
            question_type,
        )


        if (
            question_type
            not in v1[
                "by_question_type"
            ]
            or
            question_type
            not in v2[
                "by_question_type"
            ]
        ):

            continue


        old = (
            v1[
                "by_question_type"
            ][
                question_type
            ][
                "10"
            ]
        )

        new = (
            v2[
                "by_question_type"
            ][
                question_type
            ][
                "10"
            ]
        )


        for metric in (
            "recall",
            "complete",
            "mrr",
            "ndcg",
        ):

            print(
                f"{metric:10s} "
                f"V1={old[metric]:.4f} "
                f"V2={new[metric]:.4f} "
                f"delta={new[metric]-old[metric]:+.4f}"
            )


    print(
        "\nLATENCY — PROVISIONAL"
    )


    for metric in (
        "mean_ms",
        "p50_ms",
        "p95_ms",
    ):

        old = (
            v1[
                "latency"
            ][
                metric
            ]
        )

        new = (
            v2[
                "latency"
            ][
                metric
            ]
        )


        print(
            f"{metric:10s} "
            f"V1={old:.2f} "
            f"V2={new:.2f} "
            f"delta={new-old:+.2f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading frozen evaluation set..."
    )


    with open(
        EVAL_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        frozen = json.load(
            file
        )


    answerable = [

        example

        for example
        in frozen

        if example.get(
            "evidence"
        )
    ]


    print(
        "Frozen examples:",
        len(
            frozen
        )
    )

    print(
        "Answerable retrieval examples:",
        len(
            answerable
        )
    )


    # ========================================================
    # V1
    # ========================================================

    v1 = benchmark_system(
        answerable,
        SYSTEMS[
            "v1"
        ],
    )


    # Important for local Qdrant storage lock.
    gc.collect()


    # ========================================================
    # V2-A
    # ========================================================

    v2 = benchmark_system(
        answerable,
        SYSTEMS[
            "v2"
        ],
    )


    print_comparison(
        v1,
        v2,
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
                    "full_retrieval_v1_v2",

                "evaluation":
                    (
                        "Dataset Evidence Recall "
                        "on frozen answerable set"
                    ),

                "final_top_k":
                    FINAL_TOP_K,

                "v1":
                    v1,

                "v2":
                    v2,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":

    main()