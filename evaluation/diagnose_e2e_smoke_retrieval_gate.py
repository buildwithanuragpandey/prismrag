import ast
import json

from collections import Counter
from pathlib import Path

from adaptive_agentic_rag.orchestration.graph import (
    route_after_evidence,
)

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = Path(
    "evaluation/datasets/"
    "frozen_eval_500.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "adaptive_retry_frozen500_diagnostic.json"
)


MAX_RETRIES = 1


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


    if isinstance(
        payload,
        list,
    ):

        return payload


    if isinstance(
        payload,
        dict,
    ):

        for key in (
            "examples",
            "records",
            "data",
        ):

            candidate = (
                payload.get(
                    key
                )
            )


            if isinstance(
                candidate,
                list,
            ):

                return candidate


    raise ValueError(
        (
            "Unsupported frozen evaluation "
            f"dataset structure: "
            f"{type(payload).__name__}"
        )
    )


# ============================================================
# Document helpers
# ============================================================

def unique_document_ids_from_results(
    results,
):

    output = []


    for item in (
        results
        or []
    ):

        document_id = (
            item.get(
                "document_id"
            )
        )


        if (
            document_id
            and
            document_id
            not in output
        ):

            output.append(
                document_id
            )


    return output


def unique_document_ids_from_context(
    context,
):

    output = []


    if context is None:

        return output


    for item in (
        context.items
        or []
    ):

        document_id = (
            item.document_id
        )


        if (
            document_id
            and
            document_id
            not in output
        ):

            output.append(
                document_id
            )


    return output


# ============================================================
# Gold diagnostics
#
# IMPORTANT:
#
# Gold information is used ONLY for offline analysis.
# It is never used by AdaptiveRetryPolicy.
# ============================================================

def gold_recall(
    predicted_ids,
    gold_ids,
):

    gold = set(
        gold_ids
        or []
    )


    predicted = set(
        predicted_ids
        or []
    )


    if not gold:

        return None


    return (
        len(
            gold
            &
            predicted
        )
        /
        len(
            gold
        )
    )


def complete_gold(
    predicted_ids,
    gold_ids,
):

    gold = set(
        gold_ids
        or []
    )


    if not gold:

        return None


    predicted = set(
        predicted_ids
        or []
    )


    return gold.issubset(
        predicted
    )


def mean(
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


# ============================================================
# Evidence-reason telemetry
# ============================================================

def reason_value(
    reasons,
    prefix,
):

    for reason in (
        reasons
        or []
    ):

        if not isinstance(
            reason,
            str,
        ):

            continue


        if reason.startswith(
            prefix
        ):

            return (
                reason[
                    len(
                        prefix
                    ):
                ]
                .strip()
            )


    return None


def reason_list(
    reasons,
    prefix,
):

    raw_value = (
        reason_value(
            reasons=
                reasons,

            prefix=
                prefix,
        )
    )


    if raw_value is None:

        return []


    try:

        parsed = (
            ast.literal_eval(
                raw_value
            )
        )

    except (
        ValueError,
        SyntaxError,
    ):

        return []


    if not isinstance(
        parsed,
        list,
    ):

        return []


    return [
        str(
            value
        )

        for value
        in parsed
    ]


def extract_evidence_telemetry(
    reasons,
):

    reasons = list(
        reasons
        or []
    )


    return {
        "evidence_path":
            reason_value(
                reasons=
                    reasons,

                prefix=
                    "evidence_path=",
            ),

        "required_sources":
            reason_list(
                reasons=
                    reasons,

                prefix=
                    "required_sources=",
            ),

        "covered_sources":
            reason_list(
                reasons=
                    reasons,

                prefix=
                    "covered_sources=",
            ),

        "missing_sources":
            reason_list(
                reasons=
                    reasons,

                prefix=
                    "missing_sources=",
            ),
    }


# ============================================================
# Gold-side failure taxonomy
#
# This taxonomy is diagnostic only.
#
# Production routing does NOT use gold recall.
# ============================================================

def classify_gold_failure(
    *,
    retrieval_recall,
    context_recall,
    context_complete,
):

    if (
        context_complete
        is True
    ):

        return (
            "grader_reject_complete_gold_context"
        )


    if (
        context_recall
        is not None
        and
        context_recall
        >=
        0.75
    ):

        return (
            "grader_reject_high_gold_context"
        )


    if (
        retrieval_recall
        is not None
        and
        retrieval_recall
        >=
        0.75
        and
        (
            context_recall
            is None
            or
            context_recall
            <
            0.75
        )
    ):

        return (
            "context_loss_candidate"
        )


    return (
        "retrieval_miss_candidate"
    )


# ============================================================
# Production-side classification
# ============================================================

def classify_production_state(
    *,
    null_example,
    initial_sufficient,
    initial_route,
    gold_failure_category,
):

    if null_example:

        if initial_sufficient:

            return (
                "null_false_accept"
            )


        if (
            initial_route
            ==
            "rewrite"
        ):

            return (
                "null_retry_candidate"
            )


        return (
            "null_correct_reject"
        )


    if initial_sufficient:

        return (
            "initial_gate_accept"
        )


    if (
        initial_route
        ==
        "rewrite"
    ):

        return (
            "production_retry_candidate"
        )


    return (
        "production_abstain__"
        +
        str(
            gold_failure_category
        )
    )


# ============================================================
# One example
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


    question_type = (
        example[
            "question_type"
        ]
    )


    null_example = (
        question_type
        ==
        "null_query"
    )


    # ========================================================
    # Frozen-eval schema
    # ========================================================

    gold_document_ids = list(
        example.get(
            "evidence_document_ids",
            [],
        )
        or []
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
    }


    # ========================================================
    # Route query
    # ========================================================

    state.update(
        nodes.route_query(
            state
        )
    )


    router_query_type = (
        state[
            "query_type"
        ]
    )


    retrieval_strategy = (
        state[
            "retrieval_strategy"
        ]
    )


    # ========================================================
    # Retrieval attempt 0
    # ========================================================

    state.update(
        nodes.retrieve(
            state
        )
    )


    initial_retrieved_ids = (
        unique_document_ids_from_results(
            state[
                "retrieved_results"
            ]
        )
    )


    # ========================================================
    # Context attempt 0
    # ========================================================

    state.update(
        nodes.build_context(
            state
        )
    )


    initial_context_ids = (
        unique_document_ids_from_context(
            state[
                "context"
            ]
        )
    )


    # ========================================================
    # Evidence Gate attempt 0
    # ========================================================

    state.update(
        nodes.grade_evidence(
            state
        )
    )


    initial_sufficient = bool(
        state[
            "evidence_sufficient"
        ]
    )


    initial_score = (
        state[
            "evidence_score"
        ]
    )


    initial_reasons = list(
        state.get(
            "evidence_reasons",
            [],
        )
        or []
    )


    initial_telemetry = (
        extract_evidence_telemetry(
            initial_reasons
        )
    )


    # ========================================================
    # Gold diagnostics
    # ========================================================

    initial_retrieval_recall = (
        gold_recall(
            predicted_ids=
                initial_retrieved_ids,

            gold_ids=
                gold_document_ids,
        )
    )


    initial_context_recall = (
        gold_recall(
            predicted_ids=
                initial_context_ids,

            gold_ids=
                gold_document_ids,
        )
    )


    initial_context_complete = (
        complete_gold(
            predicted_ids=
                initial_context_ids,

            gold_ids=
                gold_document_ids,
        )
    )


    # ========================================================
    # Production AdaptiveRetryPolicy decision
    #
    # This is the SAME routing function used by LangGraph.
    # ========================================================

    initial_route = (
        route_after_evidence(
            state
        )
    )


    # ========================================================
    # Gold-side failure category
    # ========================================================

    gold_failure_category = None


    if (
        not null_example
        and
        not initial_sufficient
    ):

        gold_failure_category = (
            classify_gold_failure(
                retrieval_recall=
                    initial_retrieval_recall,

                context_recall=
                    initial_context_recall,

                context_complete=
                    initial_context_complete,
            )
        )


    # ========================================================
    # Production-side category
    # ========================================================

    production_category = (
        classify_production_state(
            null_example=
                null_example,

            initial_sufficient=
                initial_sufficient,

            initial_route=
                initial_route,

            gold_failure_category=
                gold_failure_category,
        )
    )


    # ========================================================
    # Defaults before optional retry
    # ========================================================

    rewrite_attempted = False

    rewritten_query = None

    final_retrieved_ids = (
        initial_retrieved_ids
    )

    final_context_ids = (
        initial_context_ids
    )

    final_retrieval_recall = (
        initial_retrieval_recall
    )

    final_context_recall = (
        initial_context_recall
    )

    final_context_complete = (
        initial_context_complete
    )

    final_sufficient = (
        initial_sufficient
    )

    final_score = (
        initial_score
    )

    final_reasons = (
        initial_reasons
    )

    final_telemetry = (
        initial_telemetry
    )


    # ========================================================
    # Policy-approved retry ONLY
    # ========================================================

    if (
        initial_route
        ==
        "rewrite"
    ):

        rewrite_attempted = True


        # ----------------------------------------------------
        # Query rewrite
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Retrieval retry
        # ----------------------------------------------------

        state.update(
            nodes.retrieve(
                state
            )
        )


        final_retrieved_ids = (
            unique_document_ids_from_results(
                state[
                    "retrieved_results"
                ]
            )
        )


        # ----------------------------------------------------
        # Context retry
        # ----------------------------------------------------

        state.update(
            nodes.build_context(
                state
            )
        )


        final_context_ids = (
            unique_document_ids_from_context(
                state[
                    "context"
                ]
            )
        )


        # ----------------------------------------------------
        # Evidence re-grade
        # ----------------------------------------------------

        state.update(
            nodes.grade_evidence(
                state
            )
        )


        final_sufficient = bool(
            state[
                "evidence_sufficient"
            ]
        )


        final_score = (
            state[
                "evidence_score"
            ]
        )


        final_reasons = list(
            state.get(
                "evidence_reasons",
                [],
            )
            or []
        )


        final_telemetry = (
            extract_evidence_telemetry(
                final_reasons
            )
        )


        # ----------------------------------------------------
        # Gold diagnostics after retry
        # ----------------------------------------------------

        final_retrieval_recall = (
            gold_recall(
                predicted_ids=
                    final_retrieved_ids,

                gold_ids=
                    gold_document_ids,
            )
        )


        final_context_recall = (
            gold_recall(
                predicted_ids=
                    final_context_ids,

                gold_ids=
                    gold_document_ids,
            )
        )


        final_context_complete = (
            complete_gold(
                predicted_ids=
                    final_context_ids,

                gold_ids=
                    gold_document_ids,
            )
        )


    # ========================================================
    # Final production route
    # ========================================================

    final_route = (
        route_after_evidence(
            state
        )
    )


    if (
        final_route
        ==
        "rewrite"
    ):

        raise RuntimeError(
            (
                "AdaptiveRetryPolicy requested another "
                "rewrite after retry budget should have "
                "been exhausted."
            )
        )


    # ========================================================
    # Deltas
    # ========================================================

    rewrite_retrieval_delta = None

    rewrite_context_delta = None

    rewrite_score_delta = None


    if rewrite_attempted:

        if (
            initial_retrieval_recall
            is not None
            and
            final_retrieval_recall
            is not None
        ):

            rewrite_retrieval_delta = (
                final_retrieval_recall
                -
                initial_retrieval_recall
            )


        if (
            initial_context_recall
            is not None
            and
            final_context_recall
            is not None
        ):

            rewrite_context_delta = (
                final_context_recall
                -
                initial_context_recall
            )


        if (
            initial_score
            is not None
            and
            final_score
            is not None
        ):

            rewrite_score_delta = (
                final_score
                -
                initial_score
            )


    # ========================================================
    # Retry outcome
    # ========================================================

    if not rewrite_attempted:

        retry_outcome = (
            "not_retried"
        )


    elif final_sufficient:

        retry_outcome = (
            "retry_rescued"
        )


    else:

        retry_outcome = (
            "retry_not_rescued"
        )


    # ========================================================
    # Record
    # ========================================================

    return {
        "id":
            example.get(
                "id"
            ),

        "question":
            question,

        "question_type":
            question_type,

        "gold_answer":
            example.get(
                "answer"
            ),

        "gold_document_ids":
            gold_document_ids,

        "is_null":
            null_example,

        # ----------------------------------------------------
        # Router
        # ----------------------------------------------------

        "router_query_type":
            router_query_type,

        "retrieval_strategy":
            retrieval_strategy,

        # ----------------------------------------------------
        # Initial retrieval/context
        # ----------------------------------------------------

        "initial_retrieved_document_ids":
            initial_retrieved_ids,

        "initial_context_document_ids":
            initial_context_ids,

        "initial_retrieval_gold_recall":
            initial_retrieval_recall,

        "initial_context_gold_recall":
            initial_context_recall,

        "initial_context_gold_complete":
            initial_context_complete,

        # ----------------------------------------------------
        # Initial evidence
        # ----------------------------------------------------

        "initial_evidence_sufficient":
            initial_sufficient,

        "initial_evidence_score":
            initial_score,

        "initial_evidence_reasons":
            initial_reasons,

        "initial_evidence_path":
            initial_telemetry[
                "evidence_path"
            ],

        "initial_required_sources":
            initial_telemetry[
                "required_sources"
            ],

        "initial_covered_sources":
            initial_telemetry[
                "covered_sources"
            ],

        "initial_missing_sources":
            initial_telemetry[
                "missing_sources"
            ],

        # ----------------------------------------------------
        # Production control policy
        # ----------------------------------------------------

        "initial_route":
            initial_route,

        "production_category":
            production_category,

        "gold_failure_category":
            gold_failure_category,

        # ----------------------------------------------------
        # Rewrite
        # ----------------------------------------------------

        "rewrite_attempted":
            rewrite_attempted,

        "rewritten_query":
            rewritten_query,

        "retry_count":
            state.get(
                "retry_count",
                0,
            ),

        "retry_outcome":
            retry_outcome,

        "rewrite_rescued":
            (
                rewrite_attempted
                and
                final_sufficient
            ),

        # ----------------------------------------------------
        # Final retrieval/context
        # ----------------------------------------------------

        "final_retrieved_document_ids":
            final_retrieved_ids,

        "final_context_document_ids":
            final_context_ids,

        "final_retrieval_gold_recall":
            final_retrieval_recall,

        "final_context_gold_recall":
            final_context_recall,

        "final_context_gold_complete":
            final_context_complete,

        # ----------------------------------------------------
        # Final evidence
        # ----------------------------------------------------

        "final_evidence_sufficient":
            final_sufficient,

        "final_evidence_score":
            final_score,

        "final_evidence_reasons":
            final_reasons,

        "final_evidence_path":
            final_telemetry[
                "evidence_path"
            ],

        "final_required_sources":
            final_telemetry[
                "required_sources"
            ],

        "final_covered_sources":
            final_telemetry[
                "covered_sources"
            ],

        "final_missing_sources":
            final_telemetry[
                "missing_sources"
            ],

        "final_route":
            final_route,

        # ----------------------------------------------------
        # Retry deltas
        # ----------------------------------------------------

        "rewrite_retrieval_recall_delta":
            rewrite_retrieval_delta,

        "rewrite_context_recall_delta":
            rewrite_context_delta,

        "rewrite_evidence_score_delta":
            rewrite_score_delta,
    }


# ============================================================
# Summary helpers
# ============================================================

def count_delta(
    records,
    key,
    direction,
):

    count = 0


    for record in records:

        value = (
            record.get(
                key
            )
        )


        if value is None:

            continue


        if (
            direction == "positive"
            and
            value > 0
        ):

            count += 1


        elif (
            direction == "negative"
            and
            value < 0
        ):

            count += 1


        elif (
            direction == "zero"
            and
            value == 0
        ):

            count += 1


    return count


def counter_dict(
    values,
):

    return dict(
        Counter(
            values
        )
    )


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

        if not record[
            "is_null"
        ]
    ]


    null_examples = [
        record

        for record
        in records

        if record[
            "is_null"
        ]
    ]


    initial_rejects_answerable = [
        record

        for record
        in answerable

        if not record[
            "initial_evidence_sufficient"
        ]
    ]


    retry_candidates = [
        record

        for record
        in records

        if (
            record[
                "initial_route"
            ]
            ==
            "rewrite"
        )
    ]


    answerable_retry_candidates = [
        record

        for record
        in retry_candidates

        if not record[
            "is_null"
        ]
    ]


    null_retry_candidates = [
        record

        for record
        in retry_candidates

        if record[
            "is_null"
        ]
    ]


    rewrite_rescues = [
        record

        for record
        in retry_candidates

        if record[
            "rewrite_rescued"
        ]
    ]


    complete_context_rejected = [
        record

        for record
        in answerable

        if (
            record[
                "initial_context_gold_complete"
            ]
            is True
            and
            not record[
                "initial_evidence_sufficient"
            ]
        )
    ]


    high_context_rejected = [
        record

        for record
        in answerable

        if (
            record[
                "initial_context_gold_recall"
            ]
            is not None
            and
            record[
                "initial_context_gold_recall"
            ]
            >=
            0.75
            and
            not record[
                "initial_evidence_sufficient"
            ]
        )
    ]


    # ========================================================
    # Missing-source diagnostics for retry candidates
    # ========================================================

    missing_source_count_distribution = (
        Counter()
    )


    missing_source_frequency = (
        Counter()
    )


    required_source_count_distribution = (
        Counter()
    )


    evidence_path_distribution = (
        Counter()
    )


    for record in (
        retry_candidates
    ):

        missing_sources = (
            record[
                "initial_missing_sources"
            ]
        )


        required_sources = (
            record[
                "initial_required_sources"
            ]
        )


        missing_source_count_distribution[
            str(
                len(
                    missing_sources
                )
            )
            +
            "_missing"
        ] += 1


        required_source_count_distribution[
            str(
                len(
                    required_sources
                )
            )
            +
            "_required"
        ] += 1


        for source in (
            missing_sources
        ):

            missing_source_frequency[
                source
            ] += 1


        evidence_path_distribution[
            str(
                record[
                    "initial_evidence_path"
                ]
            )
        ] += 1


    # ========================================================
    # Question-type route breakdown
    # ========================================================

    route_by_question_type = {}


    for question_type in sorted(
        set(
            record[
                "question_type"
            ]

            for record
            in records
        )
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


        route_by_question_type[
            question_type
        ] = {
            "count":
                len(
                    subset
                ),

            "generate":
                sum(
                    1

                    for record
                    in subset

                    if (
                        record[
                            "initial_route"
                        ]
                        ==
                        "generate"
                    )
                ),

            "rewrite":
                sum(
                    1

                    for record
                    in subset

                    if (
                        record[
                            "initial_route"
                        ]
                        ==
                        "rewrite"
                    )
                ),

            "abstain":
                sum(
                    1

                    for record
                    in subset

                    if (
                        record[
                            "initial_route"
                        ]
                        ==
                        "abstain"
                    )
                ),
        }


    # ========================================================
    # Summary
    # ========================================================

    return {
        "dataset":
            str(
                DATASET_PATH
            ),

        "total":
            total,

        "answerable":
            len(
                answerable
            ),

        "null":
            len(
                null_examples
            ),

        # ----------------------------------------------------
        # Retrieval/context baseline
        # ----------------------------------------------------

        "mean_initial_retrieval_gold_recall":
            mean(
                [
                    record[
                        "initial_retrieval_gold_recall"
                    ]

                    for record
                    in answerable
                ]
            ),

        "mean_initial_context_gold_recall":
            mean(
                [
                    record[
                        "initial_context_gold_recall"
                    ]

                    for record
                    in answerable
                ]
            ),

        # ----------------------------------------------------
        # Evidence Gate
        # ----------------------------------------------------

        "initial_gate_accepts_answerable":
            sum(
                1

                for record
                in answerable

                if record[
                    "initial_evidence_sufficient"
                ]
            ),

        "initial_gate_rejects_answerable":
            len(
                initial_rejects_answerable
            ),

        "complete_gold_context_rejected":
            len(
                complete_context_rejected
            ),

        "high_gold_context_rejected":
            len(
                high_context_rejected
            ),

        # ----------------------------------------------------
        # Production AdaptiveRetryPolicy
        # ----------------------------------------------------

        "initial_route_distribution":
            counter_dict(
                record[
                    "initial_route"
                ]

                for record
                in records
            ),

        "final_route_distribution":
            counter_dict(
                record[
                    "final_route"
                ]

                for record
                in records
            ),

        "retry_candidates":
            len(
                retry_candidates
            ),

        "retry_candidate_rate":
            (
                len(
                    retry_candidates
                )
                /
                total

                if total

                else None
            ),

        "answerable_retry_candidates":
            len(
                answerable_retry_candidates
            ),

        "null_retry_candidates":
            len(
                null_retry_candidates
            ),

        "route_by_question_type":
            route_by_question_type,

        # ----------------------------------------------------
        # Retry outcomes
        # ----------------------------------------------------

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
                    retry_candidates
                )

                if retry_candidates

                else None
            ),

        "rewrite_retrieval_improved":
            count_delta(
                retry_candidates,
                "rewrite_retrieval_recall_delta",
                "positive",
            ),

        "rewrite_retrieval_harmed":
            count_delta(
                retry_candidates,
                "rewrite_retrieval_recall_delta",
                "negative",
            ),

        "rewrite_retrieval_unchanged":
            count_delta(
                retry_candidates,
                "rewrite_retrieval_recall_delta",
                "zero",
            ),

        "rewrite_context_improved":
            count_delta(
                retry_candidates,
                "rewrite_context_recall_delta",
                "positive",
            ),

        "rewrite_context_harmed":
            count_delta(
                retry_candidates,
                "rewrite_context_recall_delta",
                "negative",
            ),

        "rewrite_context_unchanged":
            count_delta(
                retry_candidates,
                "rewrite_context_recall_delta",
                "zero",
            ),

        "mean_rewrite_retrieval_recall_delta":
            mean(
                [
                    record[
                        "rewrite_retrieval_recall_delta"
                    ]

                    for record
                    in retry_candidates
                ]
            ),

        "mean_rewrite_context_recall_delta":
            mean(
                [
                    record[
                        "rewrite_context_recall_delta"
                    ]

                    for record
                    in retry_candidates
                ]
            ),

        "mean_rewrite_evidence_score_delta":
            mean(
                [
                    record[
                        "rewrite_evidence_score_delta"
                    ]

                    for record
                    in retry_candidates
                ]
            ),

        # ----------------------------------------------------
        # Source-miss patterns
        # ----------------------------------------------------

        "retry_missing_source_count_distribution":
            dict(
                missing_source_count_distribution
            ),

        "retry_required_source_count_distribution":
            dict(
                required_source_count_distribution
            ),

        "retry_missing_source_frequency":
            dict(
                missing_source_frequency
                .most_common()
            ),

        "retry_evidence_path_distribution":
            dict(
                evidence_path_distribution
            ),

        # ----------------------------------------------------
        # Null safety
        # ----------------------------------------------------

        "null_initial_false_accepts":
            sum(
                1

                for record
                in null_examples

                if record[
                    "initial_evidence_sufficient"
                ]
            ),

        "null_initial_correct_rejects":
            sum(
                1

                for record
                in null_examples

                if not record[
                    "initial_evidence_sufficient"
                ]
            ),

        # ----------------------------------------------------
        # Diagnostic taxonomy
        # ----------------------------------------------------

        "production_category_distribution":
            counter_dict(
                record[
                    "production_category"
                ]

                for record
                in records
            ),

        "gold_failure_category_distribution":
            counter_dict(
                record[
                    "gold_failure_category"
                ]

                for record
                in initial_rejects_answerable
            ),
    }


# ============================================================
# Console output
# ============================================================

def print_retry_candidate(
    *,
    index,
    total,
    record,
):

    print(
        "\n"
        +
        "=" * 100
    )


    print(
        (
            "ADAPTIVE RETRY CANDIDATE "
            f"{index}/{total}"
        )
    )


    print(
        "=" * 100
    )


    print(
        "ID:",
        record[
            "id"
        ],
    )


    print(
        "Type:",
        record[
            "question_type"
        ],
    )


    print(
        "Question:"
    )


    print(
        record[
            "question"
        ]
    )


    print(
        "\nRequired sources:",
        record[
            "initial_required_sources"
        ],
    )


    print(
        "Covered sources:",
        record[
            "initial_covered_sources"
        ],
    )


    print(
        "Missing sources:",
        record[
            "initial_missing_sources"
        ],
    )


    print(
        "Evidence path:",
        record[
            "initial_evidence_path"
        ],
    )


    print(
        "\nInitial retrieval recall:",
        record[
            "initial_retrieval_gold_recall"
        ],
    )


    print(
        "Initial context recall:",
        record[
            "initial_context_gold_recall"
        ],
    )


    print(
        "Initial context complete:",
        record[
            "initial_context_gold_complete"
        ],
    )


    print(
        "Initial evidence:",
        (
            record[
                "initial_evidence_sufficient"
            ],
            round(
                record[
                    "initial_evidence_score"
                ],
                4,
            ),
        ),
    )


    print(
        "\nRewrite:"
    )


    print(
        record[
            "rewritten_query"
        ]
    )


    print(
        "\nFinal retrieval recall:",
        record[
            "final_retrieval_gold_recall"
        ],
    )


    print(
        "Final context recall:",
        record[
            "final_context_gold_recall"
        ],
    )


    print(
        "Final context complete:",
        record[
            "final_context_gold_complete"
        ],
    )


    print(
        "Final evidence:",
        (
            record[
                "final_evidence_sufficient"
            ],
            round(
                record[
                    "final_evidence_score"
                ],
                4,
            ),
        ),
    )


    print(
        "Final route:",
        record[
            "final_route"
        ],
    )


    print(
        "Retry outcome:",
        record[
            "retry_outcome"
        ],
    )


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
        "ADAPTIVE RETRY DIAGNOSTIC — FROZEN500"
    )


    print(
        "=" * 100
    )


    print(
        "Dataset:",
        DATASET_PATH
    )


    print(
        "Examples:",
        len(
            examples
        )
    )


    print(
        "Generation:",
        "DISABLED"
    )


    print(
        "Max retries:",
        MAX_RETRIES
    )


    nodes = (
        RAGNodes()
    )


    records = []


    retry_candidate_number = 0


    try:

        for (
            index,
            example,
        ) in enumerate(
            examples,
            start=1,
        ):

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


            # ------------------------------------------------
            # Keep 500-case output readable.
            # ------------------------------------------------

            if (
                record[
                    "rewrite_attempted"
                ]
            ):

                retry_candidate_number += 1


                print_retry_candidate(
                    index=
                        retry_candidate_number,

                    total=
                        len(
                            examples
                        ),

                    record=
                        record,
                )


            elif (
                index % 25
                ==
                0
            ):

                print(
                    (
                        f"Processed {index}/"
                        f"{len(examples)} "
                        f"| retry candidates so far: "
                        f"{retry_candidate_number}"
                    )
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


    retry_candidate_records = [
        record

        for record
        in records

        if record[
            "rewrite_attempted"
        ]
    ]


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "dataset":
                    str(
                        DATASET_PATH
                    ),

                "max_retries":
                    MAX_RETRIES,

                "generation_enabled":
                    False,

                "summary":
                    summary,

                "retry_candidates":
                    retry_candidate_records,

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