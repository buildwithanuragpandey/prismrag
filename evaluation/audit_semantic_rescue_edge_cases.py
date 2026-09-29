import json

from pathlib import Path

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)


DATASET_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)

CALIBRATION_PATH = Path(
    "evaluation/results/"
    "semantic_evidence_gate_calibration.json"
)

SAFE_POLICY_PATH = Path(
    "evaluation/results/"
    "safe_semantic_rescue_policy.json"
)

STRICT_POLICY_PATH = Path(
    "evaluation/results/"
    "strict_safe_semantic_rescue_policy.json"
)

OUTPUT_PATH = Path(
    "evaluation/results/"
    "semantic_rescue_manual_audit_cases.json"
)


MAX_PRINT_CHUNKS = 5
PRINT_TEXT_CHARS = 900


# ============================================================
# Loaders
# ============================================================

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(file)


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
        "Could not locate examples "
        "inside frozen_eval_500.json."
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
# Text helper
# ============================================================

def compact_text(
    text,
    max_chars=PRINT_TEXT_CHARS,
):

    text = (
        " ".join(
            (text or "").split()
        )
    )


    if (
        len(text)
        <=
        max_chars
    ):

        return text


    return (
        text[
            :max_chars
        ].rstrip()
        +
        "..."
    )


# ============================================================
# Context serialization
# ============================================================

def serialize_context(
    context,
    gold_document_ids,
):

    gold = set(
        gold_document_ids
        or []
    )


    output = []


    for item in context.items:

        output.append(
            {
                "citation_id":
                    item.citation_id,

                "chunk_id":
                    item.chunk_id,

                "document_id":
                    item.document_id,

                "is_gold_document": (
                    item.document_id
                    in
                    gold
                ),

                "source":
                    item.source or "",

                "title":
                    item.title or "",

                "text":
                    item.text or "",
            }
        )


    return output


# ============================================================
# Run production retrieval/context only
# ============================================================

def run_case(
    nodes,
    example,
    calibration_record,
    categories,
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


    # ========================================================
    # Re-run V2 grade for visibility.
    # ========================================================

    grade = (
        nodes.evidence_grader.grade(
            query=
                question,

            context=
                context,

            query_type=
                state[
                    "query_type"
                ],
        )
    )


    gold_document_ids = (
        example.get(
            "evidence_document_ids",
            [],
        )
    )


    context_items = (
        serialize_context(
            context=
                context,

            gold_document_ids=
                gold_document_ids,
        )
    )


    context_document_ids = list(
        dict.fromkeys(
            item[
                "document_id"
            ]

            for item
            in context_items
        )
    )


    gold = set(
        gold_document_ids
    )


    context_set = set(
        context_document_ids
    )


    context_gold_recall = None


    if gold:

        context_gold_recall = (
            len(
                gold
                &
                context_set
            )
            /
            len(
                gold
            )
        )


    semantic_requirements = (
        calibration_record.get(
            "requirements",
            [],
        )
    )


    return {
        "id":
            example[
                "id"
            ],

        "audit_categories":
            sorted(
                categories
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

        "gold_document_ids":
            gold_document_ids,

        "router_query_type":
            state.get(
                "query_type"
            ),

        "retrieval_strategy":
            state.get(
                "retrieval_strategy"
            ),

        "context_document_ids":
            context_document_ids,

        "context_gold_recall":
            context_gold_recall,

        "context_gold_complete": (
            context_gold_recall
            ==
            1.0
            if (
                context_gold_recall
                is not None
            )
            else None
        ),

        "v2_sufficient":
            grade.sufficient,

        "v2_score":
            grade.evidence_score,

        "v2_query_term_coverage":
            grade.query_term_coverage,

        "v2_reasons":
            grade.reasons,

        # ----------------------------------------------------
        # Semantic features already frozen during calibration.
        # Do NOT rescore or recalibrate them here.
        # ----------------------------------------------------

        "semantic_requirements":
            semantic_requirements,

        "context_items":
            context_items,

        # ----------------------------------------------------
        # Manual audit placeholders.
        #
        # Possible final labels:
        #
        # alternate_valid_evidence
        # partial_but_sufficient
        # insufficient_evidence
        # ambiguous
        # dataset_annotation_issue
        # ----------------------------------------------------

        "manual_label":
            None,

        "manual_notes":
            None,
    }


# ============================================================
# Console rendering
# ============================================================

def print_case(
    index,
    total,
    record,
):

    print(
        "\n"
        +
        "=" * 110
    )


    print(
        (
            f"[{index}/{total}] "
            f"{record['id']} | "
            f"{record['question_type']}"
        )
    )


    print(
        "AUDIT:",
        ", ".join(
            record[
                "audit_categories"
            ]
        ),
    )


    print(
        "=" * 110
    )


    print(
        "\nQUESTION:"
    )

    print(
        record[
            "question"
        ]
    )


    print(
        "\nGOLD ANSWER:"
    )

    print(
        record[
            "gold_answer"
        ]
    )


    print(
        "\nGOLD DOCUMENTS:"
    )

    print(
        record[
            "gold_document_ids"
        ]
    )


    print(
        "\nCONTEXT GOLD RECALL:"
    )

    print(
        record[
            "context_gold_recall"
        ]
    )


    print(
        "\nV2:"
    )

    print(
        (
            f"  sufficient="
            f"{record['v2_sufficient']}"
        )
    )

    print(
        (
            f"  score="
            f"{record['v2_score']:.4f}"
        )
    )

    print(
        (
            f"  lexical coverage="
            f"{record['v2_query_term_coverage']:.4f}"
        )
    )


    print(
        "\nSEMANTIC REQUIREMENTS:"
    )


    for (
        requirement_index,
        requirement,
    ) in enumerate(
        record[
            "semantic_requirements"
        ],
        start=1,
    ):

        print(
            (
                f"  R{requirement_index}: "
                f"{requirement.get('text')}"
            )
        )


        print(
            (
                "      best_score="
                f"{requirement.get('best_score')} | "
                "anchor_ok="
                f"{requirement.get('best_anchor_ok')} | "
                "best_doc="
                f"{requirement.get('best_document_id')}"
            )
        )


    print(
        "\nCONTEXT:"
    )


    for (
        context_index,
        item,
    ) in enumerate(
        record[
            "context_items"
        ][
            :MAX_PRINT_CHUNKS
        ],
        start=1,
    ):

        marker = (
            "GOLD"
            if (
                item[
                    "is_gold_document"
                ]
            )
            else "NON-GOLD"
        )


        print(
            "\n"
            +
            "-" * 100
        )


        print(
            (
                f"Context {context_index} | "
                f"{marker} | "
                f"{item['document_id']} | "
                f"{item['citation_id']}"
            )
        )


        print(
            "Source:",
            item[
                "source"
            ],
        )


        print(
            "Title:",
            item[
                "title"
            ],
        )


        print(
            compact_text(
                item[
                    "text"
                ]
            )
        )


# ============================================================
# Summary
# ============================================================

def summarize(
    records,
):

    categories = {}


    for record in records:

        for category in (
            record[
                "audit_categories"
            ]
        ):

            categories[
                category
            ] = (
                categories.get(
                    category,
                    0,
                )
                +
                1
            )


    recalls = [
        record[
            "context_gold_recall"
        ]

        for record
        in records

        if (
            record[
                "context_gold_recall"
            ]
            is not None
        )
    ]


    return {
        "total_cases":
            len(records),

        "categories":
            categories,

        "mean_context_gold_recall": (
            sum(recalls)
            /
            len(recalls)
            if recalls
            else None
        ),

        "zero_gold_context_cases":
            sum(
                1

                for record
                in records

                if (
                    record[
                        "context_gold_recall"
                    ]
                    ==
                    0.0
                )
            ),

        "severe_miss_cases":
            sum(
                1

                for record
                in records

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
            ),

        "high_gold_context_cases":
            sum(
                1

                for record
                in records

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
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    examples = (
        load_dataset()
    )


    example_by_id = {
        example[
            "id"
        ]:
            example

        for example
        in examples
    }


    calibration_payload = (
        load_json(
            CALIBRATION_PATH
        )
    )


    validation_records = (
        calibration_payload[
            "validation_records"
        ]
    )


    validation_by_id = {
        record[
            "id"
        ]:
            record

        for record
        in validation_records
    }


    safe_payload = (
        load_json(
            SAFE_POLICY_PATH
        )
    )


    strict_payload = (
        load_json(
            STRICT_POLICY_PATH
        )
    )


    # ========================================================
    # Original permissive rescue edge cases
    #
    # These are the cases we previously called "low gold".
    # ========================================================

    safe_low_gold_ids = set(
        safe_payload[
            "validation"
        ][
            "safe_semantic_rescue"
        ][
            "low_gold_rescue_ids"
        ]
    )


    safe_severe_ids = set(
        safe_payload[
            "validation"
        ][
            "safe_semantic_rescue"
        ][
            "severe_miss_rescue_ids"
        ]
    )


    safe_zero_ids = set(
        safe_payload[
            "validation"
        ][
            "safe_semantic_rescue"
        ][
            "zero_gold_rescue_ids"
        ]
    )


    # ========================================================
    # Strict policy's newly introduced null failure
    # ========================================================

    strict_added_null_ids = set(
        strict_payload[
            "validation"
        ][
            "strict_semantic_rescue"
        ][
            "added_null_false_accept_ids"
        ]
    )


    audit_categories = {}


    def add_category(
        ids,
        category,
    ):

        for example_id in ids:

            audit_categories.setdefault(
                example_id,
                set(),
            ).add(
                category
            )


    add_category(
        safe_low_gold_ids,
        "safe_policy_low_gold_rescue",
    )


    add_category(
        safe_severe_ids,
        "safe_policy_severe_miss_rescue",
    )


    add_category(
        safe_zero_ids,
        "safe_policy_zero_gold_rescue",
    )


    add_category(
        strict_added_null_ids,
        "strict_policy_added_null_false_accept",
    )


    audit_ids = sorted(
        audit_categories
    )


    print(
        "\nSemantic Rescue Manual Audit"
    )


    print(
        "Cases:",
        len(
            audit_ids
        )
    )


    print(
        "\nCategories:"
    )


    for category in sorted(
        {
            category

            for values
            in audit_categories.values()

            for category
            in values
        }
    ):

        count = sum(
            1

            for values
            in audit_categories.values()

            if category in values
        )


        print(
            f"  {category}: {count}"
        )


    nodes = (
        RAGNodes()
    )


    records = []


    try:

        for (
            index,
            example_id,
        ) in enumerate(
            audit_ids,
            start=1,
        ):

            if (
                example_id
                not in
                example_by_id
            ):

                raise KeyError(
                    (
                        "Missing example in frozen dataset: "
                        f"{example_id}"
                    )
                )


            if (
                example_id
                not in
                validation_by_id
            ):

                raise KeyError(
                    (
                        "Missing validation feature record: "
                        f"{example_id}"
                    )
                )


            record = (
                run_case(
                    nodes=
                        nodes,

                    example=
                        example_by_id[
                            example_id
                        ],

                    calibration_record=
                        validation_by_id[
                            example_id
                        ],

                    categories=
                        audit_categories[
                            example_id
                        ],
                )
            )


            records.append(
                record
            )


            print_case(
                index=
                    index,

                total=
                    len(
                        audit_ids
                    ),

                record=
                    record,
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
        "=" * 110
    )


    print(
        "AUDIT SUMMARY"
    )


    print(
        "=" * 110
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