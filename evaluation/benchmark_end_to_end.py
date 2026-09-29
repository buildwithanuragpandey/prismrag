import argparse
import json
import math
import time

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from adaptive_agentic_rag.orchestration.graph import (
    AdaptiveRAGGraph
)


FROZEN_EVAL_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)

OUTPUT_DIR = Path(
    "evaluation/results/end_to_end"
)


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
        * (1 - weight)
        +
        ordered[upper]
        * weight
    )


def safe_mean(
    values
):

    values = list(
        values
    )

    if not values:
        return None

    return mean(
        values
    )


# ============================================================
# Dataset
# ============================================================

def load_frozen_examples():

    with open(
        FROZEN_EVAL_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        examples = json.load(
            file
        )

    if len(examples) != 500:

        raise ValueError(
            "Expected frozen evaluation set "
            f"to contain 500 examples, got {len(examples)}."
        )

    return examples


# ============================================================
# Deterministic stratified smoke selection
# ============================================================

def select_examples(
    examples: list[dict],
    limit: int | None
):

    if limit is None:

        return examples


    if limit <= 0:

        raise ValueError(
            "--limit must be > 0"
        )


    if limit >= len(examples):

        return examples


    groups = defaultdict(
        list
    )


    for example in examples:

        groups[
            example[
                "question_type"
            ]
        ].append(
            example
        )


    for group in groups.values():

        group.sort(
            key=lambda item: item[
                "id"
            ]
        )


    types = sorted(
        groups
    )


    #
    # Give each question type at least one example
    # when the requested limit is large enough.
    #

    quotas = {
        question_type: 0
        for question_type in types
    }


    remaining = limit


    if limit >= len(types):

        for question_type in types:

            quotas[
                question_type
            ] = 1

            remaining -= 1


    #
    # Distribute remaining slots proportionally
    # to the frozen dataset distribution.
    #

    total_examples = len(
        examples
    )


    raw_allocations = {}


    for question_type in types:

        raw_allocations[
            question_type
        ] = (
            remaining
            *
            len(
                groups[
                    question_type
                ]
            )
            /
            total_examples
        )


    for question_type in types:

        additional = math.floor(
            raw_allocations[
                question_type
            ]
        )

        available = (
            len(
                groups[
                    question_type
                ]
            )
            -
            quotas[
                question_type
            ]
        )

        additional = min(
            additional,
            available
        )

        quotas[
            question_type
        ] += additional

        remaining -= additional


    #
    # Largest-remainder allocation.
    #

    remainder_order = sorted(

        types,

        key=lambda question_type: (
            raw_allocations[
                question_type
            ]
            -
            math.floor(
                raw_allocations[
                    question_type
                ]
            )
        ),

        reverse=True
    )


    while remaining > 0:

        made_progress = False


        for question_type in remainder_order:

            if (
                quotas[
                    question_type
                ]
                <
                len(
                    groups[
                        question_type
                    ]
                )
            ):

                quotas[
                    question_type
                ] += 1

                remaining -= 1

                made_progress = True


                if remaining == 0:
                    break


        if not made_progress:
            break


    selected = []


    for question_type in types:

        selected.extend(
            groups[
                question_type
            ][
                :quotas[
                    question_type
                ]
            ]
        )


    #
    # Stable ordering by ID.
    #

    selected.sort(
        key=lambda item: item[
            "id"
        ]
    )


    print(
        "Selected distribution:",
        dict(
            Counter(
                item[
                    "question_type"
                ]
                for item in selected
            )
        )
    )


    return selected


# ============================================================
# Snapshot graph state
# ============================================================

def build_record(
    example: dict,
    state: dict,
    latency_ms: float
):
    retrieved_results = (
        state.get(
            "retrieved_results"
        )
        or []
    )


    retrieved_document_ids = list(
        dict.fromkeys(
            result.get(
                "document_id"
            )

            for result
            in retrieved_results

            if result.get(
                "document_id"
            )
        )
    )


    abstained = state.get(
        "abstained",
        False
    )


    evidence_sufficient = state.get(
        "evidence_sufficient"
    )


    if not abstained:

        abstention_stage = None

    elif evidence_sufficient is False:

        abstention_stage = (
            "evidence_gate"
        )

    elif evidence_sufficient is True:

        abstention_stage = (
            "generation_or_grounding"
        )

    else:

        abstention_stage = (
            "unknown"
        )
    return {

        # ----------------------------------------------------
        # Gold / dataset information
        # ----------------------------------------------------

        "id":
            example[
                "id"
            ],

        "question":
            example[
                "question"
            ],

        "gold_answer":
            example.get(
                "answer"
            ),

        "gold_question_type":
            example[
                "question_type"
            ],

        "is_answerable":
            example[
                "is_answerable"
            ],

        "gold_evidence_document_ids":
            example.get(
                "evidence_document_ids",
                []
            ),


        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        "latency_ms":
            latency_ms,


        # ----------------------------------------------------
        # Routing
        # ----------------------------------------------------

        "predicted_query_type":
            state.get(
                "query_type"
            ),

        "retrieval_strategy":
            state.get(
                "retrieval_strategy"
            ),

        "use_reranker":
            state.get(
                "use_reranker"
            ),

        "use_mmr":
            state.get(
                "use_mmr"
            ),


        # ----------------------------------------------------
        # Self-correction
        # ----------------------------------------------------

        "original_query":
            state.get(
                "original_query"
            ),

        "current_query":
            state.get(
                "current_query"
            ),

        "retry_count":
            state.get(
                "retry_count",
                0
            ),

        "rewritten":
            state.get(
                "rewritten",
                False
            ),


        # ----------------------------------------------------
        # Retrieval/context
        # ----------------------------------------------------

        "retrieved_count":
            len(
                state.get(
                    "retrieved_results"
                )
                or []
            ),

        "evidence_sufficient":
            state.get(
                "evidence_sufficient"
            ),

        "evidence_score":
            state.get(
                "evidence_score"
            ),

        "evidence_reasons":
            state.get(
                "evidence_reasons"
            )
            or [],


        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        "abstained":
            state.get(
                "abstained",
                False
            ),

        "generation_model_name":
            state.get(
                "generation_model_name"
            ),

        "raw_answer":
            state.get(
                "raw_answer"
            ),

        "final_answer":
            state.get(
                "final_answer"
            ),


        # ----------------------------------------------------
        # Grounding diagnostics
        #
        # Important:
        # these describe the generation/grounding pipeline.
        # They are NOT treated as gold answer correctness.
        # ----------------------------------------------------

        "supported_claims":
            state.get(
                "supported_claims",
                0
            ),

        "unsupported_claims":
            state.get(
                "unsupported_claims",
                0
            ),

        "relevant_claims":
            state.get(
                "relevant_claims",
                0
            ),

        "filtered_irrelevant_claims":
            state.get(
                "filtered_irrelevant_claims",
                0
            ),


        # ----------------------------------------------------
        # Citation / internal answer grader
        # ----------------------------------------------------

        "citation_valid":
            state.get(
                "citation_valid"
            ),

        "answer_passed":
            state.get(
                "answer_passed"
            ),

        "answer_relevance_score":
            state.get(
                "answer_relevance_score"
            ),

        "answer_grade_reasons":
            state.get(
                "answer_grade_reasons"
            )
            or [],

        "retrieved_document_ids":
            retrieved_document_ids,

        "context_citations":
            extract_context_citations(
                state
            ),

        "abstention_stage":
            abstention_stage,
        # ----------------------------------------------------
        # Graph errors
        # ----------------------------------------------------

        "graph_error":
            state.get(
                "error"
            )
    }


# ============================================================
# Graph cleanup
# ============================================================

def close_graph(
    rag
):

    close_method = getattr(
        rag,
        "close",
        None
    )


    if callable(
        close_method
    ):

        close_method()

        return


    nodes = getattr(
        rag,
        "nodes",
        None
    )


    node_close = getattr(
        nodes,
        "close",
        None
    )


    if callable(
        node_close
    ):

        node_close()


# ============================================================
# Summary
# ============================================================

def build_summary(
    records: list[dict]
):

    successful = [

        record

        for record in records

        if not record.get(
            "execution_error"
        )
    ]


    errors = [

        record

        for record in records

        if record.get(
            "execution_error"
        )
    ]


    answerable = [

        record

        for record in successful

        if record[
            "is_answerable"
        ]
    ]


    null_examples = [

        record

        for record in successful

        if not record[
            "is_answerable"
        ]
    ]


    answerable_answered = [

        record

        for record in answerable

        if not record[
            "abstained"
        ]
    ]


    false_abstentions = [

        record

        for record in answerable

        if record[
            "abstained"
        ]
    ]


    correct_null_abstentions = [

        record

        for record in null_examples

        if record[
            "abstained"
        ]
    ]


    false_null_answers = [

        record

        for record in null_examples

        if not record[
            "abstained"
        ]
    ]


    rewritten = [

        record

        for record in successful

        if record[
            "rewritten"
        ]
    ]


    evidence_sufficient = [

        record

        for record in successful

        if (
            record[
                "evidence_sufficient"
            ]
            is True
        )
    ]


    answered = [

        record

        for record in successful

        if not record[
            "abstained"
        ]
    ]


    citation_valid_answered = [

        record

        for record in answered

        if (
            record[
                "citation_valid"
            ]
            is True
        )
    ]


    answer_grader_passed = [

        record

        for record in successful

        if (
            record[
                "answer_passed"
            ]
            is True
        )
    ]


    latencies = [

        record[
            "latency_ms"
        ]

        for record in successful
    ]


    route_distribution = Counter(

        (
            record[
                "predicted_query_type"
            ],
            record[
                "retrieval_strategy"
            ]
        )

        for record in successful
    )


    return {

        "total_examples":
            len(
                records
            ),

        "successful_examples":
            len(
                successful
            ),

        "execution_errors":
            len(
                errors
            ),

        "answerable_examples":
            len(
                answerable
            ),

        "null_examples":
            len(
                null_examples
            ),


        # ----------------------------------------------------
        # Evidence / rewrite
        # ----------------------------------------------------

        "evidence_sufficient_rate":
            (
                len(
                    evidence_sufficient
                )
                /
                len(
                    successful
                )

                if successful
                else None
            ),

        "rewrite_rate":
            (
                len(
                    rewritten
                )
                /
                len(
                    successful
                )

                if successful
                else None
            ),

        "avg_retry_count":
            safe_mean(
                record[
                    "retry_count"
                ]
                for record in successful
            ),


        # ----------------------------------------------------
        # Answerability behavior
        # ----------------------------------------------------

        "answerable_answer_rate":
            (
                len(
                    answerable_answered
                )
                /
                len(
                    answerable
                )

                if answerable
                else None
            ),

        "false_abstention_rate":
            (
                len(
                    false_abstentions
                )
                /
                len(
                    answerable
                )

                if answerable
                else None
            ),

        "correct_null_abstention_rate":
            (
                len(
                    correct_null_abstentions
                )
                /
                len(
                    null_examples
                )

                if null_examples
                else None
            ),

        "false_null_answer_rate":
            (
                len(
                    false_null_answers
                )
                /
                len(
                    null_examples
                )

                if null_examples
                else None
            ),


        # ----------------------------------------------------
        # Internal diagnostics
        # ----------------------------------------------------

        "citation_valid_rate_among_answered":
            (
                len(
                    citation_valid_answered
                )
                /
                len(
                    answered
                )

                if answered
                else None
            ),

        "internal_answer_grader_pass_rate":
            (
                len(
                    answer_grader_passed
                )
                /
                len(
                    successful
                )

                if successful
                else None
            ),


        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        "latency_ms":
            {

                "mean":
                    safe_mean(
                        latencies
                    ),

                "p50":
                    percentile(
                        latencies,
                        0.50
                    ),

                "p95":
                    percentile(
                        latencies,
                        0.95
                    )
            },


        "route_distribution":
            {
                f"{query_type}|{strategy}":
                    count

                for (
                    query_type,
                    strategy
                ), count
                in route_distribution.items()
            }
    }


# ============================================================
# Run benchmark
# ============================================================

def run_benchmark(
    examples: list[dict],
    max_retries: int
):

    rag = AdaptiveRAGGraph()

    records = []


    try:

        for index, example in enumerate(
            examples,
            start=1
        ):

            print(
                "\n"
                "================================"
            )

            print(
                f"{index}/{len(examples)}"
            )

            print(
                "TYPE:",
                example[
                    "question_type"
                ]
            )

            print(
                "QUESTION:",
                example[
                    "question"
                ]
            )


            start_time = time.perf_counter()


            try:

                state = rag.run(
                    query=example[
                        "question"
                    ],
                    max_retries=max_retries
                )


                latency_ms = (
                    time.perf_counter()
                    -
                    start_time
                ) * 1000.0


                record = build_record(
                    example=example,
                    state=state,
                    latency_ms=latency_ms
                )


                print(
                    "Route:",
                    record[
                        "predicted_query_type"
                    ],
                    "/",
                    record[
                        "retrieval_strategy"
                    ]
                )

                print(
                    "Evidence sufficient:",
                    record[
                        "evidence_sufficient"
                    ]
                )

                print(
                    "Rewritten:",
                    record[
                        "rewritten"
                    ]
                )

                print(
                    "Abstained:",
                    record[
                        "abstained"
                    ]
                )

                print(
                    "Answer passed:",
                    record[
                        "answer_passed"
                    ]
                )

                print(
                    "Latency ms:",
                    round(
                        latency_ms,
                        2
                    )
                )


            except Exception as error:

                latency_ms = (
                    time.perf_counter()
                    -
                    start_time
                ) * 1000.0


                record = {

                    "id":
                        example[
                            "id"
                        ],

                    "question":
                        example[
                            "question"
                        ],

                    "gold_question_type":
                        example[
                            "question_type"
                        ],

                    "is_answerable":
                        example[
                            "is_answerable"
                        ],

                    "latency_ms":
                        latency_ms,

                    "execution_error":
                        (
                            f"{type(error).__name__}: "
                            f"{error}"
                        )
                }


                print(
                    "ERROR:",
                    record[
                        "execution_error"
                    ]
                )


            records.append(
                record
            )


    finally:

        close_graph(
            rag
        )


    return records


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional stratified smoke-test limit. "
            "Omit for full frozen 500."
        )
    )


    parser.add_argument(
        "--max-retries",
        type=int,
        default=1
    )


    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()


    all_examples = (
        load_frozen_examples()
    )


    examples = (
        select_examples(
            all_examples,
            args.limit
        )
    )


    print(
        "Frozen examples:",
        len(
            all_examples
        )
    )

    print(
        "Evaluation examples:",
        len(
            examples
        )
    )


    records = run_benchmark(
        examples=examples,
        max_retries=args.max_retries
    )


    summary = build_summary(
        records
    )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    output_name = (

        f"smoke_{len(examples)}.json"

        if args.limit is not None

        else "full_500.json"
    )


    output_path = (
        OUTPUT_DIR
        /
        output_name
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "evaluation_file":
                    str(
                        FROZEN_EVAL_PATH
                    ),

                "max_retries":
                    args.max_retries,

                "summary":
                    summary,

                "records":
                    records
            },
            file,
            ensure_ascii=False,
            indent=2
        )


    print(
        "\n"
        "================================"
    )

    print(
        "END-TO-END SUMMARY"
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )

    print(
        "\nSaved:",
        output_path
    )

def extract_context_citations(
    state: dict
) -> list[dict]:

    context = state.get(
        "context"
    )

    if context is None:
        return []


    items = getattr(
        context,
        "items",
        []
    ) or []


    output = []


    for item in items:

        output.append(
            {
                "citation_id":
                    getattr(
                        item,
                        "citation_id",
                        None
                    ),

                "document_id":
                    getattr(
                        item,
                        "document_id",
                        None
                    )
            }
        )


    return output

if __name__ == "__main__":
    main()