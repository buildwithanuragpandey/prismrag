import argparse
import gc
import json
import math
import time

from collections import defaultdict
from pathlib import Path
from statistics import mean


from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever
)

from adaptive_agentic_rag.retrieval.bm25_retriever import (
    BM25Retriever
)

from adaptive_agentic_rag.retrieval.hybrid_retriever import (
    HybridRetriever
)

from adaptive_agentic_rag.retrieval.reranked_retriever import (
    RerankedRetriever
)

from adaptive_agentic_rag.retrieval.adaptive_retriever import (
    AdaptiveRetriever
)


# ============================================================
# Configuration
# ============================================================

FROZEN_EVAL_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)

OUTPUT_DIR = Path(
    "evaluation/results/frozen_retrieval"
)


KS = [
    1,
    3,
    5,
    10,
    20
]

MAX_K = max(KS)


SUPPORTED_SYSTEMS = {
    "dense",
    "bm25",
    "hybrid",
    "reranked_mmr",
    "adaptive",
}


# ============================================================
# Utilities
# ============================================================

def percentile(
    values: list[float],
    p: float
) -> float | None:

    if not values:
        return None

    ordered = sorted(
        values
    )

    if len(ordered) == 1:
        return ordered[0]

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
        return ordered[
            lower
        ]

    weight = (
        position
        -
        lower
    )

    return (
        ordered[lower]
        *
        (1 - weight)
        +
        ordered[upper]
        *
        weight
    )


def safe_float(
    value
):

    if value is None:
        return None

    try:
        return float(
            value
        )

    except (
        TypeError,
        ValueError
    ):
        return None


def json_safe(
    value
):

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool
        )
    ):
        return value

    if isinstance(
        value,
        dict
    ):
        return {
            str(key):
                json_safe(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple
        )
    ):
        return [
            json_safe(item)
            for item
            in value
        ]

    enum_value = getattr(
        value,
        "value",
        None
    )

    if enum_value is not None:
        return json_safe(
            enum_value
        )

    return str(
        value
    )


# ============================================================
# Load frozen evaluation set
# ============================================================

def load_answerable_examples():

    with open(
        FROZEN_EVAL_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        examples = json.load(
            file
        )


    answerable = [

        example

        for example in examples

        if (
            example[
                "is_answerable"
            ]
            and
            example[
                "evidence_document_ids"
            ]
        )
    ]


    print(
        "Frozen examples:",
        len(examples)
    )

    print(
        "Answerable retrieval examples:",
        len(answerable)
    )


    if (
        len(answerable)
        !=
        420
    ):

        raise ValueError(
            "Expected exactly 420 "
            "answerable retrieval examples."
        )


    return answerable


# ============================================================
# Retriever output normalization
# ============================================================

def search_retriever(
    retriever,
    query: str,
    top_k: int
):

    raw_output = (
        retriever.search(
            query,
            top_k=top_k
        )
    )


    metadata = {}


    if isinstance(
        raw_output,
        dict
    ):

        results = (
            raw_output.get(
                "results",
                []
            )
        )

        if (
            "decision"
            in raw_output
        ):

            metadata[
                "decision"
            ] = json_safe(
                raw_output[
                    "decision"
                ]
            )

    else:

        results = (
            raw_output
        )


    if results is None:
        results = []


    return (
        list(results),
        metadata
    )


# ============================================================
# Document-level metric helpers
# ============================================================

def document_ids_from_results(
    results: list[dict]
) -> list[str]:

    output = []


    for item in results:

        document_id = (
            item.get(
                "document_id"
            )
        )


        if document_id:

            output.append(
                document_id
            )


    return output


def compute_metrics_at_k(
    results: list[dict],
    relevant_document_ids: set[str],
    k: int
) -> dict:

    top_results = (
        results[
            :k
        ]
    )

    document_ids = (
        document_ids_from_results(
            top_results
        )
    )


    unique_document_ids = set(
        document_ids
    )


    retrieved_relevant = (
        unique_document_ids
        &
        relevant_document_ids
    )


    # --------------------------------------------------------
    # Recall@k
    #
    # Fraction of gold evidence documents retrieved.
    # --------------------------------------------------------

    recall = (

        len(
            retrieved_relevant
        )
        /
        len(
            relevant_document_ids
        )
    )


    # --------------------------------------------------------
    # Hit@k
    # --------------------------------------------------------

    hit = float(
        len(
            retrieved_relevant
        )
        >
        0
    )


    # --------------------------------------------------------
    # Complete Evidence Recall@k
    #
    # 1 only if ALL required supporting documents
    # have appeared by rank k.
    # --------------------------------------------------------

    complete_recall = float(

        relevant_document_ids
        .issubset(
            unique_document_ids
        )
    )


    # --------------------------------------------------------
    # MRR@k
    #
    # Rank of first relevant chunk.
    # --------------------------------------------------------

    reciprocal_rank = 0.0


    for rank, document_id in enumerate(
        document_ids,
        start=1
    ):

        if (
            document_id
            in
            relevant_document_ids
        ):

            reciprocal_rank = (
                1.0
                /
                rank
            )

            break


    # --------------------------------------------------------
    # nDCG@k
    #
    # Document-level binary relevance.
    #
    # Duplicate chunks from the same supporting document
    # do not receive relevance credit more than once.
    # --------------------------------------------------------

    seen_documents = set()

    relevance_flags = []


    for document_id in (
        document_ids
    ):

        is_new_relevant = (

            document_id
            in
            relevant_document_ids

            and

            document_id
            not in
            seen_documents
        )


        relevance_flags.append(
            1.0
            if is_new_relevant
            else 0.0
        )


        seen_documents.add(
            document_id
        )


    dcg = 0.0


    for rank, relevance in enumerate(
        relevance_flags,
        start=1
    ):

        dcg += (
            relevance
            /
            math.log2(
                rank + 1
            )
        )


    ideal_relevant_count = min(

        len(
            relevant_document_ids
        ),

        k
    )


    idcg = 0.0


    for rank in range(
        1,
        ideal_relevant_count + 1
    ):

        idcg += (
            1.0
            /
            math.log2(
                rank + 1
            )
        )


    ndcg = (

        dcg
        /
        idcg

        if idcg
        else 0.0
    )


    # --------------------------------------------------------
    # Diversity / duplication
    # --------------------------------------------------------

    unique_documents = len(
        unique_document_ids
    )


    duplicate_rate = (

        1.0
        -
        (
            unique_documents
            /
            len(
                document_ids
            )
        )

        if document_ids

        else 0.0
    )


    return {

        "recall":
            recall,

        "hit":
            hit,

        "complete_recall":
            complete_recall,

        "mrr":
            reciprocal_rank,

        "ndcg":
            ndcg,

        "unique_documents":
            unique_documents,

        "duplicate_rate":
            duplicate_rate
    }


def first_complete_evidence_rank(
    results: list[dict],
    relevant_document_ids: set[str]
) -> int | None:

    found = set()


    for rank, document_id in enumerate(
        document_ids_from_results(
            results
        ),
        start=1
    ):

        if (
            document_id
            in
            relevant_document_ids
        ):

            found.add(
                document_id
            )


        if (
            relevant_document_ids
            .issubset(
                found
            )
        ):

            return rank


    return None


# ============================================================
# Evaluate one retriever
# ============================================================

def evaluate_system(
    system_name: str,
    retriever,
    examples: list[dict],
    limit: int | None = None
):

    selected_examples = (
        examples
        if limit is None
        else examples[:limit]
    )


    print(
        "\n"
        "================================"
    )

    print(
        "SYSTEM:",
        system_name
    )

    print(
        "Examples:",
        len(
            selected_examples
        )
    )


    # --------------------------------------------------------
    # Warmup
    #
    # Not measured.
    # Helps avoid first-inference overhead dominating latency.
    # --------------------------------------------------------

    print(
        "Warmup..."
    )


    try:

        search_retriever(
            retriever=retriever,
            query="What is Amazon?",
            top_k=MAX_K
        )

    except Exception as error:

        raise RuntimeError(
            f"Warmup failed for "
            f"{system_name}: {error}"
        ) from error


    per_example = []


    for index, example in enumerate(
        selected_examples,
        start=1
    ):

        query = (
            example[
                "question"
            ]
        )


        relevant_document_ids = set(
            example[
                "evidence_document_ids"
            ]
        )


        start_time = (
            time.perf_counter()
        )


        results, metadata = (
            search_retriever(

                retriever=(
                    retriever
                ),

                query=query,

                top_k=MAX_K
            )
        )


        latency_ms = (

            time.perf_counter()
            -
            start_time

        ) * 1000.0


        metrics_by_k = {}


        for k in KS:

            metrics_by_k[
                str(k)
            ] = (
                compute_metrics_at_k(

                    results=results,

                    relevant_document_ids=(
                        relevant_document_ids
                    ),

                    k=k
                )
            )


        complete_rank = (
            first_complete_evidence_rank(

                results=results,

                relevant_document_ids=(
                    relevant_document_ids
                )
            )
        )


        retrieved_snapshot = []


        for rank, result in enumerate(
            results[
                :MAX_K
            ],
            start=1
        ):

            retrieved_snapshot.append(
                {
                    "rank":
                        rank,

                    "chunk_id":
                        result.get(
                            "id"
                        ),

                    "document_id":
                        result.get(
                            "document_id"
                        ),

                    "score":
                        safe_float(
                            result.get(
                                "score"
                            )
                        )
                }
            )


        per_example.append(
            {
                "id":
                    example[
                        "id"
                    ],

                "question":
                    query,

                "question_type":
                    example[
                        "question_type"
                    ],

                "relevant_document_ids":
                    sorted(
                        relevant_document_ids
                    ),

                "returned_count":
                    len(
                        results
                    ),

                "latency_ms":
                    latency_ms,

                "first_complete_evidence_rank":
                    complete_rank,

                "metrics":
                    metrics_by_k,

                "retrieved":
                    retrieved_snapshot,

                "metadata":
                    metadata
            }
        )


        if (
            index % 25
            ==
            0
            or
            index
            ==
            len(
                selected_examples
            )
        ):

            print(
                f"{index}/"
                f"{len(selected_examples)}"
            )


    summary = build_summary(
        system_name=system_name,
        records=per_example
    )


    return {
        "system":
            system_name,

        "evaluation_file":
            str(
                FROZEN_EVAL_PATH
            ),

        "evaluated_examples":
            len(
                selected_examples
            ),

        "ks":
            KS,

        "summary":
            summary,

        "per_example":
            per_example
    }


# ============================================================
# Aggregation
# ============================================================

def aggregate_records(
    records: list[dict]
) -> dict:

    if not records:
        return {}


    output = {
        "examples":
            len(records),

        "avg_returned_count":
            mean(
                record[
                    "returned_count"
                ]
                for record
                in records
            )
    }


    latency_values = [

        record[
            "latency_ms"
        ]

        for record
        in records

    ]


    output[
        "latency_ms"
    ] = {

        "mean":
            mean(
                latency_values
            ),

        "p50":
            percentile(
                latency_values,
                0.50
            ),

        "p95":
            percentile(
                latency_values,
                0.95
            )
    }


    completed_ranks = [

        record[
            "first_complete_evidence_rank"
        ]

        for record
        in records

        if (
            record[
                "first_complete_evidence_rank"
            ]
            is not None
        )
    ]


    output[
        "complete_evidence"
    ] = {

        "completion_rate_within_returned_results":

            (
                len(
                    completed_ranks
                )
                /
                len(
                    records
                )
            ),

        "avg_first_complete_rank_when_completed":

            (
                mean(
                    completed_ranks
                )

                if completed_ranks

                else None
            )
    }


    output[
        "metrics"
    ] = {}


    metric_names = [
        "recall",
        "hit",
        "complete_recall",
        "mrr",
        "ndcg",
        "unique_documents",
        "duplicate_rate"
    ]


    for k in KS:

        k_key = str(
            k
        )

        output[
            "metrics"
        ][
            k_key
        ] = {}


        for metric_name in metric_names:

            values = [

                record[
                    "metrics"
                ][
                    k_key
                ][
                    metric_name
                ]

                for record
                in records
            ]


            output[
                "metrics"
            ][
                k_key
            ][
                metric_name
            ] = mean(
                values
            )


    return output


def build_summary(
    system_name: str,
    records: list[dict]
) -> dict:

    by_type = defaultdict(
        list
    )


    for record in records:

        by_type[
            record[
                "question_type"
            ]
        ].append(
            record
        )


    return {

        "system":
            system_name,

        "overall":
            aggregate_records(
                records
            ),

        "by_question_type":
            {

                question_type:
                    aggregate_records(
                        type_records
                    )

                for (
                    question_type,
                    type_records
                )

                in sorted(
                    by_type.items()
                )
            }
    }


# ============================================================
# Save
# ============================================================

def save_result(
    result: dict
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    system_name = (
        result[
            "system"
        ]
    )


    output_path = (

        OUTPUT_DIR
        /
        f"{system_name}.json"
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2
        )


    print(
        "Saved:",
        output_path
    )


# ============================================================
# Print compact summary
# ============================================================

def print_summary(
    result: dict
):

    overall = (
        result[
            "summary"
        ][
            "overall"
        ]
    )


    print(
        "\n===== SUMMARY ====="
    )


    for k in KS:

        metrics = (
            overall[
                "metrics"
            ][
                str(k)
            ]
        )


        print(
            f"k={k}"
            f" | Recall={metrics['recall']:.4f}"
            f" | Hit={metrics['hit']:.4f}"
            f" | Complete={metrics['complete_recall']:.4f}"
            f" | MRR={metrics['mrr']:.4f}"
            f" | nDCG={metrics['ndcg']:.4f}"
            f" | Dup={metrics['duplicate_rate']:.4f}"
        )


    latency = (
        overall[
            "latency_ms"
        ]
    )


    print(
        "\nLatency ms:"
    )

    print(
        "Mean:",
        round(
            latency[
                "mean"
            ],
            2
        )
    )

    print(
        "P50:",
        round(
            latency[
                "p50"
            ],
            2
        )
    )

    print(
        "P95:",
        round(
            latency[
                "p95"
            ],
            2
        )
    )


# ============================================================
# Cleanup
# ============================================================

def close_retriever(
    retriever
):

    close_method = getattr(
        retriever,
        "close",
        None
    )


    if callable(
        close_method
    ):

        close_method()

        return


    store = getattr(
        retriever,
        "store",
        None
    )


    client = getattr(
        store,
        "client",
        None
    )


    client_close = getattr(
        client,
        "close",
        None
    )


    if callable(
        client_close
    ):

        client_close()


# ============================================================
# Shared-model systems
# ============================================================

def run_shared_dense_systems(
    requested_systems: set[str],
    examples: list[dict],
    limit: int | None
):

    systems_needing_dense = (
        requested_systems
        &
        {
            "dense",
            "hybrid",
            "reranked_mmr"
        }
    )


    if not systems_needing_dense:
        return


    print(
        "\nLoading shared DenseRetriever..."
    )


    dense = (
        DenseRetriever()
    )


    hybrid = None

    reranked = None


    try:

        if (
            "dense"
            in
            requested_systems
        ):

            result = (
                evaluate_system(
                    system_name="dense",
                    retriever=dense,
                    examples=examples,
                    limit=limit
                )
            )

            save_result(
                result
            )

            print_summary(
                result
            )


        if (
            "hybrid"
            in
            requested_systems
        ):

            hybrid = (
                HybridRetriever(

                    dense_retriever=dense,

                    dense_top_k=MAX_K,

                    bm25_top_k=MAX_K,

                    final_top_k=MAX_K
                )
            )


            result = (
                evaluate_system(
                    system_name="hybrid",
                    retriever=hybrid,
                    examples=examples,
                    limit=limit
                )
            )

            save_result(
                result
            )

            print_summary(
                result
            )


        if (
            "reranked_mmr"
            in
            requested_systems
        ):

            #
            # This class currently includes
            # reranking + MMR.
            #
            # We intentionally label the result
            # accordingly.
            #

            reranked = (
                RerankedRetriever(

                    dense_retriever=dense,

                    hybrid_top_k=20,

                    rerank_top_k=20,

                    final_top_k=MAX_K,

                    mmr_lambda=0.7
                )
            )


            result = (
                evaluate_system(
                    system_name=(
                        "reranked_mmr"
                    ),
                    retriever=reranked,
                    examples=examples,
                    limit=limit
                )
            )

            save_result(
                result
            )

            print_summary(
                result
            )


    finally:

        #
        # Hybrid/Reranked share DenseRetriever.
        # Close the shared Dense only once.
        #

        close_retriever(
            dense
        )


        del reranked
        del hybrid
        del dense


        gc.collect()


# ============================================================
# BM25
# ============================================================

def run_bm25(
    examples: list[dict],
    limit: int | None
):

    retriever = (
        BM25Retriever()
    )


    try:

        result = (
            evaluate_system(
                system_name="bm25",
                retriever=retriever,
                examples=examples,
                limit=limit
            )
        )


        save_result(
            result
        )


        print_summary(
            result
        )


    finally:

        close_retriever(
            retriever
        )

        del retriever

        gc.collect()


# ============================================================
# Adaptive
# ============================================================

def run_adaptive(
    examples: list[dict],
    limit: int | None
):

    #
    # Important:
    #
    # AdaptiveRetriever creates its own shared
    # DenseRetriever internally.
    #
    # Run this only after the previous local Qdrant
    # client has been closed.
    #

    retriever = (
        AdaptiveRetriever()
    )


    try:

        result = (
            evaluate_system(
                system_name="adaptive",
                retriever=retriever,
                examples=examples,
                limit=limit
            )
        )


        save_result(
            result
        )


        print_summary(
            result
        )


    finally:

        close_retriever(
            retriever
        )

        del retriever

        gc.collect()


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = (
        argparse.ArgumentParser()
    )


    parser.add_argument(
        "--systems",
        nargs="+",
        default=[
            "dense",
            "bm25",
            "hybrid",
            "reranked_mmr",
            "adaptive"
        ],
        help=(
            "Systems to evaluate. "
            "Available: "
            + ", ".join(
                sorted(
                    SUPPORTED_SYSTEMS
                )
            )
        )
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. "
            "Omit for the full 420 examples."
        )
    )


    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = (
        parse_args()
    )


    requested_systems = set(
        args.systems
    )


    unknown = (
        requested_systems
        -
        SUPPORTED_SYSTEMS
    )


    if unknown:

        raise ValueError(
            "Unknown systems: "
            f"{sorted(unknown)}"
        )


    examples = (
        load_answerable_examples()
    )


    if (
        args.limit
        is not None
    ):

        if (
            args.limit
            <=
            0
        ):

            raise ValueError(
                "--limit must be > 0"
            )


        print(
            "SMOKE TEST LIMIT:",
            args.limit
        )


    # --------------------------------------------------------
    # First:
    # systems that can share one Qdrant/Dense instance.
    # --------------------------------------------------------

    run_shared_dense_systems(

        requested_systems=(
            requested_systems
        ),

        examples=examples,

        limit=args.limit
    )


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    if (
        "bm25"
        in
        requested_systems
    ):

        run_bm25(
            examples=examples,
            limit=args.limit
        )


    # --------------------------------------------------------
    # Adaptive last.
    #
    # It owns a separate local Qdrant connection.
    # --------------------------------------------------------

    if (
        "adaptive"
        in
        requested_systems
    ):

        run_adaptive(
            examples=examples,
            limit=args.limit
        )


    print(
        "\n"
        "================================"
    )

    print(
        "FROZEN RETRIEVAL BENCHMARK COMPLETE"
    )


if __name__ == "__main__":

    main()