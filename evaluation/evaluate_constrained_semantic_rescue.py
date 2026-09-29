import json

from pathlib import Path

from adaptive_agentic_rag.orchestration.constrained_semantic_rescue import (
    ConstrainedSemanticRescue,
)

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)


DATASET_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)

SPLIT_PATH = Path(
    "evaluation/datasets/"
    "semantic_gate_calibration_split.json"
)

AUDIT_PATH = Path(
    "evaluation/results/"
    "semantic_rescue_manual_audit_cases.json"
)

OUTPUT_PATH = Path(
    "evaluation/results/"
    "constrained_semantic_rescue_ablation.json"
)


# ============================================================
# Loaders
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


    for key in (
        "examples",
        "records",
        "data",
    ):

        if key in payload:

            return payload[
                key
            ]


    raise ValueError(
        "Could not locate examples."
    )


def load_json(
    path,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# Retrieval/context gold recall
# ============================================================

def gold_recall(
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


    return (
        len(
            predicted
            &
            gold
        )
        /
        len(
            gold
        )
    )


# ============================================================
# Run one example
# ============================================================

def run_case(
    nodes,
    rescue,
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
    }


    state.update(
        nodes.route_query(
            state
        )
    )


    state.update(
        nodes.retrieve(
            state
        )
    )


    state.update(
        nodes.build_context(
            state
        )
    )


    context = (
        state[
            "context"
        ]
    )


    query_type = (
        state[
            "query_type"
        ]
    )


    v2 = (
        nodes.evidence_grader.grade(
            query=
                question,

            context=
                context,

            query_type=
                query_type,
        )
    )


    semantic = (
        rescue.analyze(
            query=
                question,

            context=
                context,

            query_type=
                query_type,
        )
    )


    # ========================================================
    # Rescue is ONLY a fallback.
    #
    # V2 remains authoritative when it already accepts.
    # ========================================================

    rescued = (
        not v2.sufficient
        and
        semantic[
            "sufficient"
        ]
    )


    final_sufficient = (
        v2.sufficient
        or
        rescued
    )


    context_document_ids = list(
        dict.fromkeys(
            item.document_id

            for item
            in context.items
        )
    )


    gold_document_ids = (
        example.get(
            "evidence_document_ids",
            [],
        )
    )


    context_recall = (
        gold_recall(
            predicted_ids=
                context_document_ids,

            gold_ids=
                gold_document_ids,
        )
    )


    return {
        "id":
            example[
                "id"
            ],

        "question":
            question,

        "question_type":
            example[
                "question_type"
            ],

        "router_query_type":
            query_type,

        "gold_answer":
            example.get(
                "answer"
            ),

        "gold_document_ids":
            gold_document_ids,

        "context_document_ids":
            context_document_ids,

        "context_gold_recall":
            context_recall,

        "context_gold_complete": (
            context_recall
            ==
            1.0
            if (
                context_recall
                is not None
            )
            else None
        ),

        "v2_sufficient":
            v2.sufficient,

        "v2_score":
            v2.evidence_score,

        "semantic_rescue_sufficient":
            semantic[
                "sufficient"
            ],

        "rescued":
            rescued,

        "final_sufficient":
            final_sufficient,

        "semantic":
            semantic,
    }


# ============================================================
# Aggregate evaluation
# ============================================================

def evaluate(
    records,
):

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


    v2_answerable_accepts = sum(
        1

        for record
        in answerable

        if record[
            "v2_sufficient"
        ]
    )


    final_answerable_accepts = sum(
        1

        for record
        in answerable

        if record[
            "final_sufficient"
        ]
    )


    answerable_rescues = [
        record

        for record
        in answerable

        if record[
            "rescued"
        ]
    ]


    v2_null_false_accepts = sum(
        1

        for record
        in null_examples

        if record[
            "v2_sufficient"
        ]
    )


    final_null_false_accepts = sum(
        1

        for record
        in null_examples

        if record[
            "final_sufficient"
        ]
    )


    added_null_false_accepts = [
        record[
            "id"
        ]

        for record
        in null_examples

        if (
            not record[
                "v2_sufficient"
            ]

            and

            record[
                "final_sufficient"
            ]
        )
    ]


    complete_gold_rescues = [
        record

        for record
        in answerable_rescues

        if (
            record[
                "context_gold_recall"
            ]
            ==
            1.0
        )
    ]


    high_gold_rescues = [
        record

        for record
        in answerable_rescues

        if (
            record[
                "context_gold_recall"
            ]
            is not None

            and

            record[
                "context_gold_recall"
            ]
            >=
            0.75
        )
    ]


    low_gold_rescues = [
        record

        for record
        in answerable_rescues

        if (
            record[
                "context_gold_recall"
            ]
            is not None

            and

            record[
                "context_gold_recall"
            ]
            <
            0.75
        )
    ]


    severe_miss_rescues = [
        record

        for record
        in answerable_rescues

        if (
            record[
                "context_gold_recall"
            ]
            is not None

            and

            record[
                "context_gold_recall"
            ]
            <=
            0.25
        )
    ]


    zero_gold_rescues = [
        record

        for record
        in answerable_rescues

        if (
            record[
                "context_gold_recall"
            ]
            ==
            0.0
        )
    ]


    complete_gold_rejected = sum(
        1

        for record
        in answerable

        if (
            record[
                "context_gold_complete"
            ]
            is True

            and

            not record[
                "final_sufficient"
            ]
        )
    )


    high_gold_rejected = sum(
        1

        for record
        in answerable

        if (
            record[
                "context_gold_recall"
            ]
            is not None

            and

            record[
                "context_gold_recall"
            ]
            >=
            0.75

            and

            not record[
                "final_sufficient"
            ]
        )
    )


    by_question_type = {}


    for question_type in (
        "inference_query",
        "comparison_query",
        "temporal_query",
        "null_query",
    ):

        group = [
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


        if not group:

            continue


        by_question_type[
            question_type
        ] = {
            "count":
                len(
                    group
                ),

            "v2_accepts":
                sum(
                    1

                    for record
                    in group

                    if record[
                        "v2_sufficient"
                    ]
                ),

            "final_accepts":
                sum(
                    1

                    for record
                    in group

                    if record[
                        "final_sufficient"
                    ]
                ),

            "rescues":
                sum(
                    1

                    for record
                    in group

                    if record[
                        "rescued"
                    ]
                ),
        }


    return {
        "answerable_total":
            len(
                answerable
            ),

        "v2_answerable_accepts":
            v2_answerable_accepts,

        "v2_answerable_accept_rate": (
            v2_answerable_accepts
            /
            len(
                answerable
            )
        ),

        "final_answerable_accepts":
            final_answerable_accepts,

        "final_answerable_accept_rate": (
            final_answerable_accepts
            /
            len(
                answerable
            )
        ),

        "answerable_rescue_count":
            len(
                answerable_rescues
            ),

        "answerable_rescue_ids": [
            record[
                "id"
            ]

            for record
            in answerable_rescues
        ],

        "complete_gold_rescues":
            len(
                complete_gold_rescues
            ),

        "high_gold_rescues":
            len(
                high_gold_rescues
            ),

        "low_gold_rescues":
            len(
                low_gold_rescues
            ),

        "severe_miss_rescues":
            len(
                severe_miss_rescues
            ),

        "severe_miss_rescue_ids": [
            record[
                "id"
            ]

            for record
            in severe_miss_rescues
        ],

        "zero_gold_rescues":
            len(
                zero_gold_rescues
            ),

        "zero_gold_rescue_ids": [
            record[
                "id"
            ]

            for record
            in zero_gold_rescues
        ],

        "null_total":
            len(
                null_examples
            ),

        "v2_null_false_accepts":
            v2_null_false_accepts,

        "final_null_false_accepts":
            final_null_false_accepts,

        "added_null_false_accept_count":
            len(
                added_null_false_accepts
            ),

        "added_null_false_accept_ids":
            added_null_false_accepts,

        "complete_gold_context_rejected":
            complete_gold_rejected,

        "high_gold_context_rejected":
            high_gold_rejected,

        "by_question_type":
            by_question_type,
    }


# ============================================================
# Audit regression
# ============================================================

def build_audit_regression(
    validation_records,
):

    if not (
        AUDIT_PATH.exists()
    ):

        return {
            "available":
                False,
        }


    audit_payload = (
        load_json(
            AUDIT_PATH
        )
    )


    audit_categories = {
        record[
            "id"
        ]:
            record[
                "audit_categories"
            ]

        for record
        in audit_payload[
            "records"
        ]
    }


    validation_by_id = {
        record[
            "id"
        ]:
            record

        for record
        in validation_records
    }


    output = []


    for (
        example_id,
        categories,
    ) in sorted(
        audit_categories.items()
    ):

        record = (
            validation_by_id.get(
                example_id
            )
        )


        if record is None:

            continue


        semantic = (
            record[
                "semantic"
            ]
        )


        output.append(
            {
                "id":
                    example_id,

                "categories":
                    categories,

                "question_type":
                    record[
                        "question_type"
                    ],

                "context_gold_recall":
                    record[
                        "context_gold_recall"
                    ],

                "v2_sufficient":
                    record[
                        "v2_sufficient"
                    ],

                "rescued":
                    record[
                        "rescued"
                    ],

                "final_sufficient":
                    record[
                        "final_sufficient"
                    ],

                "supported_requirement_count":
                    semantic[
                        "supported_requirement_count"
                    ],

                "required_requirement_count":
                    semantic[
                        "required_requirement_count"
                    ],

                "document_diversity_required":
                    semantic[
                        "document_diversity_required"
                    ],

                "document_diversity_ok":
                    semantic[
                        "document_diversity_ok"
                    ],

                "supporting_document_ids":
                    semantic[
                        "supporting_document_ids"
                    ],

                "requirements": [
                    {
                        "text":
                            requirement[
                                "text"
                            ],

                        "supported":
                            requirement[
                                "supported"
                            ],

                        "best_score":
                            requirement[
                                "best_score"
                            ],

                        "best_document_id":
                            requirement[
                                "best_document_id"
                            ],

                        "best_source":
                            requirement[
                                "best_source"
                            ],

                        "explicit_sources":
                            requirement[
                                "explicit_sources"
                            ],

                        "eligible_candidate_count":
                            requirement[
                                "eligible_candidate_count"
                            ],

                        "absence_claim":
                            requirement[
                                "absence_claim"
                            ],

                        "multi_source_requirement":
                            requirement[
                                "multi_source_requirement"
                            ],
                    }

                    for requirement
                    in semantic[
                        "requirements"
                    ]
                ],
            }
        )


    return {
        "available":
            True,

        "total":
            len(
                output
            ),

        "rescued":
            sum(
                1

                for record
                in output

                if record[
                    "rescued"
                ]
            ),

        "rejected":
            sum(
                1

                for record
                in output

                if not record[
                    "final_sufficient"
                ]
            ),

        "records":
            output,
    }


# ============================================================
# Pretty print
# ============================================================

def print_metrics(
    label,
    metrics,
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
        "=" * 100
    )


    print(
        (
            "V2 answerable: "
            f"{metrics['v2_answerable_accepts']}/"
            f"{metrics['answerable_total']} "
            f"({metrics['v2_answerable_accept_rate']:.4f})"
        )
    )


    print(
        (
            "Constrained final answerable: "
            f"{metrics['final_answerable_accepts']}/"
            f"{metrics['answerable_total']} "
            f"({metrics['final_answerable_accept_rate']:.4f})"
        )
    )


    print(
        (
            "Rescued answerable: "
            f"{metrics['answerable_rescue_count']}"
        )
    )


    print(
        (
            "  complete-gold rescues: "
            f"{metrics['complete_gold_rescues']}"
        )
    )


    print(
        (
            "  high-gold rescues: "
            f"{metrics['high_gold_rescues']}"
        )
    )


    print(
        (
            "  low-gold rescues: "
            f"{metrics['low_gold_rescues']}"
        )
    )


    print(
        (
            "  severe-miss rescues: "
            f"{metrics['severe_miss_rescues']}"
        )
    )


    print(
        (
            "  zero-gold rescues: "
            f"{metrics['zero_gold_rescues']}"
        )
    )


    print(
        (
            "V2 null false accepts: "
            f"{metrics['v2_null_false_accepts']}"
        )
    )


    print(
        (
            "Final null false accepts: "
            f"{metrics['final_null_false_accepts']}"
        )
    )


    print(
        (
            "Added null false accepts: "
            f"{metrics['added_null_false_accept_count']}"
        )
    )


    print(
        (
            "Complete-gold rejected: "
            f"{metrics['complete_gold_context_rejected']}"
        )
    )


    print(
        (
            "High-gold rejected: "
            f"{metrics['high_gold_context_rejected']}"
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    examples = (
        load_examples()
    )


    examples_by_id = {
        example[
            "id"
        ]:
            example

        for example
        in examples
    }


    split = (
        load_json(
            SPLIT_PATH
        )
    )


    calibration_examples = [
        examples_by_id[
            example_id
        ]

        for example_id
        in split[
            "calibration_ids"
        ]
    ]


    validation_examples = [
        examples_by_id[
            example_id
        ]

        for example_id
        in split[
            "validation_ids"
        ]
    ]


    print(
        "Calibration examples:",
        len(
            calibration_examples
        )
    )


    print(
        "Validation examples:",
        len(
            validation_examples
        )
    )


    nodes = (
        RAGNodes()
    )


    rescue = (
        ConstrainedSemanticRescue(
            reranker=(
                nodes
                .retriever
                .reranked
                .reranker
            ),

            evidence_grader=(
                nodes
                .evidence_grader
            ),
        )
    )


    calibration_records = []

    validation_records = []


    try:

        # ====================================================
        # No threshold selection happens here.
        #
        # Calibration is evaluated only as a sanity check.
        # ====================================================

        for (
            index,
            example,
        ) in enumerate(
            calibration_examples,
            start=1,
        ):

            print(
                (
                    f"[CAL] "
                    f"{index}/"
                    f"{len(calibration_examples)} "
                    f"{example['question_type']} "
                    f"{example['id']}"
                )
            )


            calibration_records.append(
                run_case(
                    nodes=
                        nodes,

                    rescue=
                        rescue,

                    example=
                        example,
                )
            )


        # ====================================================
        # Frozen held-out validation.
        # ====================================================

        for (
            index,
            example,
        ) in enumerate(
            validation_examples,
            start=1,
        ):

            print(
                (
                    f"[VAL] "
                    f"{index}/"
                    f"{len(validation_examples)} "
                    f"{example['question_type']} "
                    f"{example['id']}"
                )
            )


            validation_records.append(
                run_case(
                    nodes=
                        nodes,

                    rescue=
                        rescue,

                    example=
                        example,
                )
            )


    finally:

        nodes.close()


    calibration_metrics = (
        evaluate(
            calibration_records
        )
    )


    validation_metrics = (
        evaluate(
            validation_records
        )
    )


    audit_regression = (
        build_audit_regression(
            validation_records
        )
    )


    print_metrics(
        "CALIBRATION",
        calibration_metrics,
    )


    print_metrics(
        "HELD-OUT VALIDATION",
        validation_metrics,
    )


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "AUDIT REGRESSION"
    )


    print(
        "=" * 100
    )


    if not (
        audit_regression[
            "available"
        ]
    ):

        print(
            "Audit file not found."
        )


    else:

        print(
            "Cases:",
            audit_regression[
                "total"
            ],
        )


        print(
            "Still rescued:",
            audit_regression[
                "rescued"
            ],
        )


        print(
            "Rejected:",
            audit_regression[
                "rejected"
            ],
        )


        for record in (
            audit_regression[
                "records"
            ]
        ):

            print(
                "\n"
                +
                "-" * 90
            )


            print(
                (
                    f"{record['id']} | "
                    f"gold_recall="
                    f"{record['context_gold_recall']} | "
                    f"rescued="
                    f"{record['rescued']} | "
                    f"final="
                    f"{record['final_sufficient']}"
                )
            )


            print(
                "categories:",
                record[
                    "categories"
                ],
            )


            print(
                (
                    "requirements: "
                    f"{record['supported_requirement_count']}/"
                    f"{record['required_requirement_count']}"
                )
            )


            print(
                (
                    "diversity: "
                    f"required="
                    f"{record['document_diversity_required']} "
                    f"ok="
                    f"{record['document_diversity_ok']}"
                )
            )


            print(
                (
                    "supporting docs:",
                    record[
                        "supporting_document_ids"
                    ],
                )
            )


            for requirement in (
                record[
                    "requirements"
                ]
            ):

                print(
                    (
                        "  - "
                        f"supported="
                        f"{requirement['supported']} | "
                        f"score="
                        f"{requirement['best_score']} | "
                        f"doc="
                        f"{requirement['best_document_id']} | "
                        f"source="
                        f"{requirement['best_source']} | "
                        f"eligible="
                        f"{requirement['eligible_candidate_count']} | "
                        f"absence="
                        f"{requirement['absence_claim']} | "
                        f"multi_source="
                        f"{requirement['multi_source_requirement']}"
                    )
                )


                print(
                    (
                        "    "
                        f"{requirement['text']}"
                    )
                )


    output = {
        "policy": {
            "threshold":
                rescue.threshold,

            "required_fraction":
                rescue.required_fraction,

            "structural_constraints": [
                "explicit_source_binding",
                "local_anchor_filter_before_reranking",
                "multi_source_requirement_guard",
                "absence_claim_guard",
                "comparison_document_diversity",
            ],
        },

        "calibration": {
            "metrics":
                calibration_metrics,

            "records":
                calibration_records,
        },

        "validation": {
            "metrics":
                validation_metrics,

            "records":
                validation_records,
        },

        "audit_regression":
            audit_regression,
    }


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
            output,
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