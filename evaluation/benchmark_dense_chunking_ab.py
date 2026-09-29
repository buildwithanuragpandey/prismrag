import json
import math
import statistics
import time
from pathlib import Path


from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever,
)


# ============================================================
# Configuration
# ============================================================

EVAL_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)

OUTPUT_PATH = Path(
    "evaluation/results/"
    "dense_chunking_ab.json"
)


V1_COLLECTION = (
    "multihop_chunks"
)

V2_COLLECTION = (
    "multihop_chunks_v2"
)


K_VALUES = (
    1,
    3,
    5,
    10,
    20,
)


MAX_K = max(
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
        len(ordered)
        -
        1
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
            ordered[
                lower
            ]
        )


    fraction = (
        position
        -
        lower
    )


    return float(

        ordered[
            lower
        ]
        *
        (
            1
            -
            fraction
        )

        +

        ordered[
            upper
        ]
        *
        fraction
    )


# ============================================================
# Metrics
# ============================================================

def evaluate_at_k(
    results,
    gold_document_ids,
    k,
):

    selected = (
        results[
            :k
        ]
    )


    retrieved_document_ids = [

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
            retrieved_document_ids
        )
    )


    gold = set(
        gold_document_ids
    )


    retrieved_gold = (
        gold
        &
        set(
            unique_documents
        )
    )


    # ========================================================
    # Recall
    #
    # Document-level evidence recall.
    # ========================================================

    recall = (

        len(
            retrieved_gold
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
            retrieved_gold
        )
    )


    complete = float(
        bool(
            gold
        )
        and
        gold.issubset(
            set(
                unique_documents
            )
        )
    )


    # ========================================================
    # MRR
    #
    # First relevant document occurrence.
    # ========================================================

    reciprocal_rank = 0.0


    for rank, document_id in enumerate(
        retrieved_document_ids,
        start=1,
    ):

        if document_id in gold:

            reciprocal_rank = (
                1.0
                /
                rank
            )

            break


    # ========================================================
    # nDCG
    #
    # Binary document relevance.
    #
    # Duplicate chunks from the same document do NOT receive
    # relevance credit multiple times.
    # ========================================================

    dcg = 0.0

    already_seen = set()


    for rank, document_id in enumerate(
        retrieved_document_ids,
        start=1,
    ):

        relevance = 0.0


        if (
            document_id in gold
            and
            document_id not in already_seen
        ):

            relevance = 1.0


        already_seen.add(
            document_id
        )


        if relevance:

            dcg += (
                relevance
                /
                math.log2(
                    rank
                    +
                    1
                )
            )


    ideal_relevant_count = min(
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
            ideal_relevant_count
            +
            1,
        )
    )


    ndcg = (
        dcg
        /
        idcg

        if idcg > 0

        else 0.0
    )


    # ========================================================
    # Diversity
    # ========================================================

    result_count = len(
        retrieved_document_ids
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
            reciprocal_rank,

        "ndcg":
            ndcg,

        "unique_documents":
            unique_count,

        "duplicate_rate":
            duplicate_rate,
    }


# ============================================================
# Benchmark one collection
# ============================================================

def benchmark_collection(
    examples,
    collection_name,
    label,
):

    print(
        "\n"
        +
        "=" * 100
    )

    print(
        label
    )

    print(
        "Collection:",
        collection_name
    )

    print(
        "=" * 100
    )


    retriever = DenseRetriever(
        collection_name=
            collection_name
    )


    metric_values = {

        k: {
            "recall": [],
            "hit": [],
            "complete": [],
            "mrr": [],
            "ndcg": [],
            "unique_documents": [],
            "duplicate_rate": [],
        }

        for k in K_VALUES
    }


    latencies = []

    per_example = []


    try:

        total = len(
            examples
        )


        for index, example in enumerate(
            examples,
            start=1,
        ):

            question = (
                example[
                    "question"
                ]
            )


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


            start_time = (
                time.perf_counter()
            )


            results = (
                retriever.search(
                    question,
                    top_k=MAX_K,
                )
            )


            latency_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0


            latencies.append(
                latency_ms
            )


            example_metrics = {}


            for k in K_VALUES:

                metrics = (
                    evaluate_at_k(
                        results,
                        gold_document_ids,
                        k,
                    )
                )


                example_metrics[
                    str(
                        k
                    )
                ] = metrics


                for metric_name, value in (
                    metrics.items()
                ):

                    metric_values[
                        k
                    ][
                        metric_name
                    ].append(
                        value
                    )


            per_example.append(
                {
                    "id":
                        example[
                            "id"
                        ],

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

                    "metrics":
                        example_metrics,
                }
            )


            if (
                index % 50 == 0
                or
                index == total
            ):

                print(
                    f"{index}/{total}"
                )


    finally:

        retriever.close()


    summary = {}


    for k in K_VALUES:

        values = (
            metric_values[
                k
            ]
        )


        summary[
            str(
                k
            )
        ] = {

            metric_name:
                float(
                    statistics.mean(
                        metric_list
                    )
                )

            for metric_name, metric_list
            in values.items()
        }


    latency_summary = {

        "mean_ms":
            float(
                statistics.mean(
                    latencies
                )
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

        "min_ms":
            min(
                latencies
            ),

        "max_ms":
            max(
                latencies
            ),
    }


    return {
        "label":
            label,

        "collection":
            collection_name,

        "summary":
            summary,

        "latency":
            latency_summary,

        "per_example":
            per_example,
    }


# ============================================================
# Print summary
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
        "DENSE CHUNKING A/B"
    )

    print(
        "=" * 110
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

            delta = (
                new
                -
                old
            )


            print(
                f"{metric:16s} "
                f"V1={old:.4f} "
                f"V2={new:.4f} "
                f"delta={delta:+.4f}"
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

        examples = json.load(
            file
        )


    # ========================================================
    # Retrieval benchmark evaluates answerable examples only.
    #
    # Null examples have no evidence documents.
    # ========================================================

    answerable = [

        example

        for example
        in examples

        if example.get(
            "evidence"
        )
    ]


    print(
        "Frozen examples:",
        len(
            examples
        )
    )

    print(
        "Answerable retrieval examples:",
        len(
            answerable
        )
    )


    # This should currently be 420.
    if not answerable:

        raise RuntimeError(
            "No answerable evaluation examples."
        )


    # ========================================================
    # IMPORTANT:
    #
    # Run sequentially.
    #
    # Both collections are stored in the same local Qdrant
    # directory and we do not want two local clients holding
    # the storage lock simultaneously.
    # ========================================================

    v1 = benchmark_collection(

        examples=
            answerable,

        collection_name=
            V1_COLLECTION,

        label=
            "V1 — 2000 char chunks",
    )


    v2 = benchmark_collection(

        examples=
            answerable,

        collection_name=
            V2_COLLECTION,

        label=
            "V2-A — 1000 char chunks",
    )


    print_comparison(
        v1,
        v2,
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    payload = {
        "experiment":
            "dense_chunking_ab",

        "hypothesis":
            (
                "Reducing chunk size from 2000 to "
                "1000 characters reduces semantic "
                "dilution and improves dense "
                "document retrieval."
            ),

        "controlled_variables":
            {
                "embedding_model":
                    "Qwen/Qwen3-Embedding-0.6B",

                "v1_embedding_representation":
                    "chunk text",

                "v2_embedding_representation":
                    "chunk text",

                "v1_collection":
                    V1_COLLECTION,

                "v2_collection":
                    V2_COLLECTION,
            },

        "evaluation_examples":
            len(
                answerable
            ),

        "v1":
            v1,

        "v2":
            v2,
    }


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
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