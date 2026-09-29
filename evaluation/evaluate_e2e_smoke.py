import json
import math
import re
import time

from collections import Counter
from pathlib import Path

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)

from adaptive_agentic_rag.orchestration.graph import (
    route_after_evidence,
)


MAX_RETRIES = 1


DATASET_PATH = Path(
    "evaluation/datasets/"
    "frozen_e2e_smoke_20.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "e2e_smoke_20_results.json"
)


# ============================================================
# Dataset
# ============================================================

def load_examples():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(
            file
        )


    if not isinstance(
        payload,
        list,
    ):

        raise ValueError(
            "Expected frozen_e2e_smoke_20.json "
            "to contain a list."
        )


    return payload


# ============================================================
# Normalization / smoke accuracy
#
# IMPORTANT:
#
# This is intentionally lightweight.
#
# It is NOT the final answer-quality evaluator.
# ============================================================

def normalize_text(
    value,
):

    if value is None:

        return ""


    text = (
        str(
            value
        )
        .lower()
        .strip()
    )


    text = re.sub(
        r"\[\d+\]",
        " ",
        text,
    )


    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )


    text = re.sub(
        r"\s+",
        " ",
        text,
    )


    return (
        text.strip()
    )


def yes_no_label(
    value,
):

    normalized = (
        normalize_text(
            value
        )
    )


    if not normalized:

        return None


    first = (
        normalized.split()[
            0
        ]
    )


    if first == "yes":

        return "yes"


    if first == "no":

        return "no"


    return None


def token_f1(
    prediction,
    gold,
):

    prediction_tokens = (
        normalize_text(
            prediction
        )
        .split()
    )


    gold_tokens = (
        normalize_text(
            gold
        )
        .split()
    )


    if (
        not prediction_tokens
        or
        not gold_tokens
    ):

        return 0.0


    prediction_counter = (
        Counter(
            prediction_tokens
        )
    )


    gold_counter = (
        Counter(
            gold_tokens
        )
    )


    overlap = sum(
        (
            prediction_counter
            &
            gold_counter
        ).values()
    )


    if overlap == 0:

        return 0.0


    precision = (
        overlap
        /
        len(
            prediction_tokens
        )
    )


    recall = (
        overlap
        /
        len(
            gold_tokens
        )
    )


    return (
        2
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )
    )


def smoke_answer_correct(
    prediction,
    gold,
):

    if gold is None:

        return None


    gold_text = (
        normalize_text(
            gold
        )
    )


    prediction_text = (
        normalize_text(
            prediction
        )
    )


    if not gold_text:

        return None


    if not prediction_text:

        return False


    # ========================================================
    # Boolean answer
    # ========================================================

    gold_boolean = (
        yes_no_label(
            gold
        )
    )


    if gold_boolean:

        return (
            yes_no_label(
                prediction
            )
            ==
            gold_boolean
        )


    # ========================================================
    # Exact
    # ========================================================

    if (
        prediction_text
        ==
        gold_text
    ):

        return True


    # ========================================================
    # Short-span containment
    # ========================================================

    gold_tokens = (
        gold_text.split()
    )


    if (
        len(
            gold_tokens
        )
        <=
        12
        and
        gold_text
        in
        prediction_text
    ):

        return True


    # ========================================================
    # Smoke-only free-form heuristic
    # ========================================================

    return (
        token_f1(
            prediction=
                prediction,

            gold=
                gold,
        )
        >=
        0.70
    )


# ============================================================
# Utility
# ============================================================

def safe_mean(
    values,
):

    values = [
        value

        for value
        in values

        if value is not None
    ]


    if not values:

        return None


    return (
        sum(
            values
        )
        /
        len(
            values
        )
    )


def percentile(
    values,
    percentile_value,
):

    values = sorted(
        value

        for value
        in values

        if value is not None
    )


    if not values:

        return None


    if (
        len(
            values
        )
        ==
        1
    ):

        return values[
            0
        ]


    position = (
        (
            len(
                values
            )
            -
            1
        )
        *
        percentile_value
    )


    lower = (
        math.floor(
            position
        )
    )


    upper = (
        math.ceil(
            position
        )
    )


    if lower == upper:

        return values[
            lower
        ]


    weight = (
        position
        -
        lower
    )


    return (
        values[
            lower
        ]
        *
        (
            1
            -
            weight
        )
        +
        values[
            upper
        ]
        *
        weight
    )


def is_null_example(
    example,
):

    return (
        example.get(
            "question_type"
        )
        ==
        "null_query"
    )


# ============================================================
# Citation → document mapping
# ============================================================

def citation_document_ids(
    generation_result,
    context,
):

    citation_map = {
        item.citation_id:
            item.document_id

        for item
        in context.items
    }


    result = []


    for citation_id in (
        generation_result
        .cited_ids
    ):

        document_id = (
            citation_map.get(
                citation_id
            )
        )


        if (
            document_id
            is not None
        ):

            result.append(
                document_id
            )


    return list(
        dict.fromkeys(
            result
        )
    )


def dataset_citation_metrics(
    cited_document_ids,
    gold_evidence_ids,
):

    gold = set(
        gold_evidence_ids
        or []
    )


    cited = set(
        cited_document_ids
        or []
    )


    if not gold:

        return {
            "precision":
                None,

            "recall":
                None,
        }


    intersection = (
        gold
        &
        cited
    )


    precision = (
        len(
            intersection
        )
        /
        len(
            cited
        )
        if cited
        else 0.0
    )


    recall = (
        len(
            intersection
        )
        /
        len(
            gold
        )
    )


    return {
        "precision":
            precision,

        "recall":
            recall,
    }


# ============================================================
# Production-path execution
# ============================================================

def run_example(
    nodes,
    example,
):

    question = (
        example[
            "question"
        ]
    )


    state = {
        "original_query":
            question,

        "current_query":
            question,

        "retry_count":
            0,

        "max_retries":
            MAX_RETRIES,

        "retry_target_sources":
            [],
    }


    case_started = (
        time.perf_counter()
    )


    # ========================================================
    # Route
    # ========================================================

    route_started = (
        time.perf_counter()
    )


    state.update(
        nodes.route_query(
            state
        )
    )


    route_seconds = (
        time.perf_counter()
        -
        route_started
    )


    # ========================================================
    # Retrieval attempt 0
    # ========================================================

    retrieval_started = (
        time.perf_counter()
    )


    state.update(
        nodes.retrieve(
            state
        )
    )


    retrieval_seconds = (
        time.perf_counter()
        -
        retrieval_started
    )


    state.update(
        nodes.build_context(
            state
        )
    )


    state.update(
        nodes.grade_evidence(
            state
        )
    )


    initial_evidence_sufficient = (
        state[
            "evidence_sufficient"
        ]
    )


    initial_evidence_score = (
        state[
            "evidence_score"
        ]
    )


    initial_query = (
        state[
            "current_query"
        ]
    )


    # ========================================================
    # Production-aligned adaptive retry routing
    #
    # route_after_evidence() is the single source of truth.
    # ========================================================

    initial_evidence_route = (
        route_after_evidence(
            state
        )
    )


    rewrite_attempted = False

    rewrite_rescued = False

    rewritten_query = None

    rewrite_retrieval_seconds = 0.0


    # ========================================================
    # Policy-approved self-correction
    # ========================================================

    if (
        initial_evidence_route
        ==
        "rewrite"
    ):

        rewrite_attempted = True


        state.update(
            nodes.rewrite_query(
                state
            )
        )


        rewritten_query = (
            state[
                "current_query"
            ]
        )


        retry_started = (
            time.perf_counter()
        )


        state.update(
            nodes.retrieve(
                state
            )
        )


        rewrite_retrieval_seconds = (
            time.perf_counter()
            -
            retry_started
        )


        state.update(
            nodes.build_context(
                state
            )
        )


        state.update(
            nodes.grade_evidence(
                state
            )
        )


        rewrite_rescued = bool(
            state[
                "evidence_sufficient"
            ]
        )


    elif (
        initial_evidence_route
        in {
            "generate",
            "abstain",
        }
    ):

        pass


    else:

        raise RuntimeError(
            (
                "Unsupported initial evidence route: "
                f"{initial_evidence_route!r}"
            )
        )


    final_evidence_sufficient = (
        state[
            "evidence_sufficient"
        ]
    )


    final_evidence_score = (
        state[
            "evidence_score"
        ]
    )


    final_evidence_route = (
        route_after_evidence(
            state
        )
    )


    if (
        final_evidence_route
        ==
        "rewrite"
    ):

        raise RuntimeError(
            (
                "Adaptive retry policy requested another "
                "rewrite after the configured retry budget."
            )
        )


    # ========================================================
    # Production generator
    # ========================================================

    generation_started = (
        time.perf_counter()
    )


    generation_result = (
        nodes.generator.generate(
            query=
                question,

            context=
                state[
                    "context"
                ],

            evidence_sufficient=
                final_evidence_sufficient,
        )
    )


    generation_seconds = (
        time.perf_counter()
        -
        generation_started
    )


    # ========================================================
    # Runtime AnswerGrader
    # ========================================================

    answer_grade = (
        nodes.answer_grader.grade(
            query=
                question,

            generation_result=
                generation_result,

            evidence_sufficient=
                final_evidence_sufficient,
        )
    )


    case_seconds = (
        time.perf_counter()
        -
        case_started
    )


    # ========================================================
    # Dataset evidence citations
    # ========================================================

    cited_document_ids = (
        citation_document_ids(
            generation_result=
                generation_result,

            context=
                state[
                    "context"
                ],
        )
    )


    citation_metrics = (
        dataset_citation_metrics(
            cited_document_ids=
                cited_document_ids,

            gold_evidence_ids=
                example.get(
                    "evidence_document_ids",
                    [],
                ),
        )
    )


    null_example = (
        is_null_example(
            example
        )
    )


    smoke_correct = None


    if not null_example:

        if (
            generation_result
            .abstained
        ):

            smoke_correct = False


        else:

            smoke_correct = (
                smoke_answer_correct(
                    prediction=(
                        generation_result
                        .direct_answer
                        or
                        generation_result
                        .answer
                    ),

                    gold=
                        example.get(
                            "answer"
                        ),
                )
            )


    # ========================================================
    # Evaluation record
    # ========================================================

    return {
        "id":
            example.get(
                "id"
            ),

        "question":
            question,

        "question_type":
            example.get(
                "question_type"
            ),

        "gold_answer":
            example.get(
                "answer"
            ),

        "gold_evidence_ids":
            example.get(
                "evidence_document_ids",
                [],
            ),

        # ----------------------------------------------------
        # Routing
        # ----------------------------------------------------

        "router_query_type":
            state.get(
                "query_type"
            ),

        "retrieval_strategy":
            state.get(
                "retrieval_strategy"
            ),

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        "initial_query":
            initial_query,

        "initial_evidence_sufficient":
            initial_evidence_sufficient,

        "initial_evidence_score":
            initial_evidence_score,

        "initial_evidence_route":
            initial_evidence_route,

        "rewrite_attempted":
            rewrite_attempted,

        "rewritten_query":
            rewritten_query,

        "rewrite_rescued":
            rewrite_rescued,

        "final_evidence_sufficient":
            final_evidence_sufficient,

        "final_evidence_score":
            final_evidence_score,

        "final_evidence_route":
            final_evidence_route,

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        "answer":
            generation_result
            .answer,

        "direct_answer":
            generation_result
            .direct_answer,

        # IMPORTANT:
        # Preserve Qwen's draft direct answer BEFORE
        # RelationAwareAnswerResolver potentially changes it.
        "draft_direct_answer":
            getattr(
                generation_result,
                "draft_direct_answer",
                None,
            ),

        "raw_answer":
            generation_result
            .raw_answer,

        "abstained":
            generation_result
            .abstained,

        "draft_claims":
            generation_result
            .draft_claims,

        "supported_claims":
            generation_result
            .supported_claims,

        "unsupported_claims":
            generation_result
            .unsupported_claims,

        "relevant_claims":
            generation_result
            .relevant_claims,

        "filtered_irrelevant_claims":
            generation_result
            .filtered_irrelevant_claims,

        # ----------------------------------------------------
        # Citations
        # ----------------------------------------------------

        "citation_valid":
            generation_result
            .citation_valid,

        "cited_ids":
            generation_result
            .cited_ids,

        "cited_document_ids":
            cited_document_ids,

        "dataset_evidence_citation_precision":
            citation_metrics[
                "precision"
            ],

        "dataset_evidence_citation_recall":
            citation_metrics[
                "recall"
            ],

        # ----------------------------------------------------
        # Runtime grader
        # ----------------------------------------------------

        "runtime_grader_passed":
            answer_grade
            .passed,

        "runtime_supported_claim_ratio":
            answer_grade
            .supported_claim_ratio,

        "runtime_relevance_score":
            answer_grade
            .relevance_score,

        "runtime_reasons":
            answer_grade
            .reasons,

        # ----------------------------------------------------
        # Smoke answer metric
        # ----------------------------------------------------

        "smoke_answer_correct":
            smoke_correct,

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        "route_seconds":
            route_seconds,

        "initial_retrieval_seconds":
            retrieval_seconds,

        "rewrite_retrieval_seconds":
            rewrite_retrieval_seconds,

        "generation_seconds":
            generation_seconds,

        "case_seconds":
            case_seconds,
    }


# ============================================================
# Summary
# ============================================================

def summarize(
    records,
):

    total = (
        len(
            records
        )
    )


    answerable = [
        record

        for record
        in records

        if (
            record[
                "question_type"
            ]
            !=
            "null_query"
        )
    ]


    null_examples = [
        record

        for record
        in records

        if (
            record[
                "question_type"
            ]
            ==
            "null_query"
        )
    ]


    answered_answerable = [
        record

        for record
        in answerable

        if not (
            record[
                "abstained"
            ]
        )
    ]


    false_abstentions = [
        record

        for record
        in answerable

        if (
            record[
                "abstained"
            ]
        )
    ]


    null_abstentions = [
        record

        for record
        in null_examples

        if (
            record[
                "abstained"
            ]
        )
    ]


    rewrite_attempts = [
        record

        for record
        in records

        if (
            record[
                "rewrite_attempted"
            ]
        )
    ]


    rewrite_rescues = [
        record

        for record
        in rewrite_attempts

        if (
            record[
                "rewrite_rescued"
            ]
        )
    ]


    smoke_scored = [
        record

        for record
        in answerable

        if (
            record[
                "smoke_answer_correct"
            ]
            is not None
        )
    ]


    smoke_correct = [
        record

        for record
        in smoke_scored

        if (
            record[
                "smoke_answer_correct"
            ]
        )
    ]


    runtime_passes = [
        record

        for record
        in records

        if (
            record[
                "runtime_grader_passed"
            ]
        )
    ]


    citation_valid_answered = [
        record

        for record
        in answered_answerable

        if (
            record[
                "citation_valid"
            ]
        )
    ]


    citation_precision_values = [
        record[
            "dataset_evidence_citation_precision"
        ]

        for record
        in answered_answerable

        if (
            record[
                "dataset_evidence_citation_precision"
            ]
            is not None
        )
    ]


    citation_recall_values = [
        record[
            "dataset_evidence_citation_recall"
        ]

        for record
        in answered_answerable

        if (
            record[
                "dataset_evidence_citation_recall"
            ]
            is not None
        )
    ]


    case_times = [
        record[
            "case_seconds"
        ]

        for record
        in records
    ]


    generation_records = [
        record

        for record
        in records

        if (
            record[
                "generation_seconds"
            ]
            >
            0
        )
    ]


    generation_times = [
        record[
            "generation_seconds"
        ]

        for record
        in generation_records
    ]


    # ========================================================
    # First generator invocation includes lazy model load.
    # ========================================================

    warm_generation_times = (
        generation_times[
            1:
        ]
        if (
            len(
                generation_times
            )
            >
            1
        )
        else []
    )


    # ========================================================
    # Runtime unsupported proxy
    # ========================================================

    answered_runtime_failures = [
        record

        for record
        in answered_answerable

        if not (
            record[
                "runtime_grader_passed"
            ]
        )
    ]


    # ========================================================
    # Question-type breakdown
    # ========================================================

    by_question_type = {}


    question_types = sorted(
        set(
            record[
                "question_type"
            ]

            for record
            in records
        )
    )


    for question_type in (
        question_types
    ):

        subset = [
            record

            for record
            in records

            if (
                record[
                    "question_type"
                ]
                ==
                question_type
            )
        ]


        answered = [
            record

            for record
            in subset

            if not (
                record[
                    "abstained"
                ]
            )
        ]


        correct = [
            record

            for record
            in subset

            if (
                record[
                    "smoke_answer_correct"
                ]
                is True
            )
        ]


        scored = [
            record

            for record
            in subset

            if (
                record[
                    "smoke_answer_correct"
                ]
                is not None
            )
        ]


        by_question_type[
            question_type
        ] = {
            "count":
                len(
                    subset
                ),

            "answered":
                len(
                    answered
                ),

            "abstained":
                (
                    len(
                        subset
                    )
                    -
                    len(
                        answered
                    )
                ),

            "smoke_scored":
                len(
                    scored
                ),

            "smoke_correct":
                len(
                    correct
                ),

            "smoke_accuracy":
                (
                    len(
                        correct
                    )
                    /
                    len(
                        scored
                    )

                    if scored
                    else None
                ),
        }


    initial_route_distribution = (
        Counter(
            record[
                "initial_evidence_route"
            ]

            for record
            in records
        )
    )


    final_route_distribution = (
        Counter(
            record[
                "final_evidence_route"
            ]

            for record
            in records
        )
    )


    return {
        "dataset":
            str(
                DATASET_PATH
            ),

        "total_cases":
            total,

        "answerable_cases":
            len(
                answerable
            ),

        "null_cases":
            len(
                null_examples
            ),

        # ----------------------------------------------------
        # Answer / abstention
        # ----------------------------------------------------

        "answerable_answered":
            len(
                answered_answerable
            ),

        "answer_rate":
            (
                len(
                    answered_answerable
                )
                /
                len(
                    answerable
                )

                if answerable
                else None
            ),

        "false_abstentions":
            len(
                false_abstentions
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

        "null_abstentions":
            len(
                null_abstentions
            ),

        "null_abstention_rate":
            (
                len(
                    null_abstentions
                )
                /
                len(
                    null_examples
                )

                if null_examples
                else None
            ),

        # ----------------------------------------------------
        # Smoke answer correctness
        # ----------------------------------------------------

        "smoke_answer_scored":
            len(
                smoke_scored
            ),

        "smoke_answer_correct":
            len(
                smoke_correct
            ),

        "smoke_answer_accuracy":
            (
                len(
                    smoke_correct
                )
                /
                len(
                    smoke_scored
                )

                if smoke_scored
                else None
            ),

        # ----------------------------------------------------
        # Runtime grader
        # ----------------------------------------------------

        "runtime_grader_passes":
            len(
                runtime_passes
            ),

        "runtime_grader_pass_rate":
            (
                len(
                    runtime_passes
                )
                /
                total

                if total
                else None
            ),

        "answered_runtime_failures":
            len(
                answered_runtime_failures
            ),

        "runtime_unsupported_proxy_rate":
            (
                len(
                    answered_runtime_failures
                )
                /
                len(
                    answered_answerable
                )

                if answered_answerable
                else None
            ),

        # ----------------------------------------------------
        # Citations
        # ----------------------------------------------------

        "citation_valid_answered":
            len(
                citation_valid_answered
            ),

        "citation_validity_rate":
            (
                len(
                    citation_valid_answered
                )
                /
                len(
                    answered_answerable
                )

                if answered_answerable
                else None
            ),

        "mean_dataset_evidence_citation_precision":
            safe_mean(
                citation_precision_values
            ),

        "mean_dataset_evidence_citation_recall":
            safe_mean(
                citation_recall_values
            ),

        # ----------------------------------------------------
        # Rewrite
        # ----------------------------------------------------

        "rewrite_attempts":
            len(
                rewrite_attempts
            ),

        "rewrite_rate":
            (
                len(
                    rewrite_attempts
                )
                /
                total

                if total
                else None
            ),

        "rewrite_rescues":
            len(
                rewrite_rescues
            ),

        "rewrite_rescue_rate":
            (
                len(
                    rewrite_rescues
                )
                /
                len(
                    rewrite_attempts
                )

                if rewrite_attempts
                else None
            ),

        # ----------------------------------------------------
        # Adaptive retry routing
        # ----------------------------------------------------

        "initial_evidence_route_distribution":
            dict(
                initial_route_distribution
            ),

        "final_evidence_route_distribution":
            dict(
                final_route_distribution
            ),

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        "mean_case_seconds":
            safe_mean(
                case_times
            ),

        "p95_case_seconds":
            percentile(
                case_times,
                0.95,
            ),

        "mean_generation_seconds_including_cold_start":
            safe_mean(
                generation_times
            ),

        "mean_generation_seconds_warm_provisional":
            safe_mean(
                warm_generation_times
            ),

        "p95_generation_seconds_warm_provisional":
            percentile(
                warm_generation_times,
                0.95,
            ),

        # ----------------------------------------------------
        # Breakdown
        # ----------------------------------------------------

        "by_question_type":
            by_question_type,
    }


# ============================================================
# Main
# ============================================================

def main():

    examples = (
        load_examples()
    )


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "E2E SMOKE BENCHMARK"
    )


    print(
        "=" * 100
    )


    print(
        "Examples:",
        len(
            examples
        )
    )


    nodes = (
        RAGNodes()
    )


    records = []


    try:

        for (
            index,
            example,
        ) in enumerate(
            examples,
            start=1,
        ):

            print(
                "\n"
                +
                "=" * 100
            )


            print(
                (
                    f"[{index}/"
                    f"{len(examples)}] "
                    f"{example.get('question_type')}"
                )
            )


            print(
                "=" * 100
            )


            print(
                "QUESTION:"
            )


            print(
                example[
                    "question"
                ]
            )


            print(
                "\nGOLD:"
            )


            print(
                example.get(
                    "answer"
                )
            )


            record = (
                run_example(
                    nodes=
                        nodes,

                    example=
                        example,
                )
            )


            records.append(
                record
            )


            print(
                "\n===== RESULT ====="
            )


            print(
                "Initial evidence sufficient:",
                record[
                    "initial_evidence_sufficient"
                ],
            )


            print(
                "Initial evidence score:",
                round(
                    record[
                        "initial_evidence_score"
                    ],
                    4,
                ),
            )


            print(
                "Initial evidence route:",
                record[
                    "initial_evidence_route"
                ],
            )


            print(
                "Rewrite attempted:",
                record[
                    "rewrite_attempted"
                ],
            )


            if (
                record[
                    "rewrite_attempted"
                ]
            ):

                print(
                    "Rewritten query:",
                    record[
                        "rewritten_query"
                    ],
                )


                print(
                    "Rewrite rescued:",
                    record[
                        "rewrite_rescued"
                    ],
                )


            print(
                "Final evidence sufficient:",
                record[
                    "final_evidence_sufficient"
                ],
            )


            print(
                "Final evidence score:",
                round(
                    record[
                        "final_evidence_score"
                    ],
                    4,
                ),
            )


            print(
                "Final evidence route:",
                record[
                    "final_evidence_route"
                ],
            )


            print(
                "\nANSWER:"
            )


            print(
                record[
                    "answer"
                ]
            )


            print(
                "\nDraft direct answer:",
                record[
                    "draft_direct_answer"
                ],
            )


            print(
                "Final direct answer:",
                record[
                    "direct_answer"
                ],
            )


            print(
                "\nAbstained:",
                record[
                    "abstained"
                ],
            )


            print(
                "Smoke answer correct:",
                record[
                    "smoke_answer_correct"
                ],
            )


            print(
                "Runtime grader passed:",
                record[
                    "runtime_grader_passed"
                ],
            )


            print(
                "Citation valid:",
                record[
                    "citation_valid"
                ],
            )


            print(
                (
                    "Dataset Evidence Citation "
                    "Precision:"
                ),
                record[
                    "dataset_evidence_citation_precision"
                ],
            )


            print(
                (
                    "Dataset Evidence Citation "
                    "Recall:"
                ),
                record[
                    "dataset_evidence_citation_recall"
                ],
            )


            print(
                "Case seconds:",
                round(
                    record[
                        "case_seconds"
                    ],
                    3,
                ),
            )


    finally:

        nodes.close()


    summary = (
        summarize(
            records
        )
    )


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "SUMMARY"
    )


    print(
        "=" * 100
    )


    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
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
                "max_retries":
                    MAX_RETRIES,

                "summary":
                    summary,

                "records":
                    records,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


    print(
        "\nSaved:"
    )


    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":

    main()