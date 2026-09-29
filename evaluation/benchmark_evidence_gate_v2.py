import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from adaptive_agentic_rag.agents.query_router import (
    QueryRouter,
)

from adaptive_agentic_rag.agents.query_rewriter import (
    QueryRewriter,
)

from adaptive_agentic_rag.agents.evidence_grader import (
    EvidenceGrader,
)

from adaptive_agentic_rag.retrieval.adaptive_retriever import (
    AdaptiveRetriever,
)

from adaptive_agentic_rag.generation.context_builder import (
    ContextBuilder,
)


FROZEN_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)

OUTPUT_PATH = Path(
    "evaluation/results/"
    "evidence_gate_v2_500.json"
)


COLLECTION_NAME = (
    "multihop_chunks_v2"
)

BM25_CORPUS_PATH = (
    "data/processed/"
    "processed_corpus_v2.json"
)

TOP_K = 10

MAX_RETRIES = 1


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

        seen.add(value)
        output.append(value)

    return output


def gold_document_ids(
    example,
):

    return unique_preserve_order(

        evidence["document_id"]

        for evidence
        in example.get(
            "evidence",
            []
        )
    )


def result_document_ids(
    results,
):

    return unique_preserve_order(

        result.get(
            "document_id"
        )

        for result
        in results
    )


def context_document_ids(
    context,
):

    if context is None:
        return []

    items = (
        getattr(
            context,
            "items",
            []
        )
        or []
    )

    return unique_preserve_order(

        getattr(
            item,
            "document_id",
            None
        )

        for item
        in items
    )


def document_recall(
    predicted,
    gold,
):

    if not gold:
        return None

    return (
        len(
            set(predicted)
            &
            set(gold)
        )
        /
        len(
            set(gold)
        )
    )


def extract_low_term_coverage(
    reasons,
):

    pattern = re.compile(
        (
            r"Query term coverage is too low:\s*"
            r"([0-9.]+)\s*<\s*([0-9.]+)"
        )
    )

    for reason in (
        reasons
        or []
    ):

        match = pattern.search(
            reason
        )

        if match:

            return {
                "observed":
                    float(
                        match.group(1)
                    ),

                "threshold":
                    float(
                        match.group(2)
                    ),
            }

    return None


# ============================================================
# One attempt
# ============================================================

def run_attempt(
    *,
    retriever,
    context_builder,
    evidence_grader,
    original_query,
    retrieval_query,
    query_type,
    gold_docs,
):

    output = retriever.search(
        retrieval_query,
        top_k=TOP_K,
    )

    results = output[
        "results"
    ]


    retrieved_docs = (
        result_document_ids(
            results
        )
    )


    context = (
        context_builder.build(
            results,
            query=original_query,
        )
    )


    context_docs = (
        context_document_ids(
            context
        )
    )


    grade = (
        evidence_grader.grade(
            query=original_query,
            context=context,
            query_type=query_type,
        )
    )


    reasons = list(
        grade.reasons
        or []
    )


    return {
        "retrieval_query":
            retrieval_query,

        "retrieved_document_ids":
            retrieved_docs,

        "context_document_ids":
            context_docs,

        "retrieved_gold_recall":
            document_recall(
                retrieved_docs,
                gold_docs,
            ),

        "context_gold_recall":
            document_recall(
                context_docs,
                gold_docs,
            ),

        "evidence_sufficient":
            bool(
                grade.sufficient
            ),

        "evidence_score":
            float(
                grade.evidence_score
            ),

        "evidence_reasons":
            reasons,

        "low_term_coverage":
            extract_low_term_coverage(
                reasons
            ),
    }


# ============================================================
# Benchmark
# ============================================================

def run_benchmark(
    examples,
):

    retriever = AdaptiveRetriever(
        collection_name=
            COLLECTION_NAME,

        bm25_corpus_path=
            BM25_CORPUS_PATH,
    )

    router = QueryRouter()

    rewriter = QueryRewriter()

    context_builder = (
        ContextBuilder()
    )

    evidence_grader = (
        EvidenceGrader()
    )


    records = []


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


            gold_docs = (
                gold_document_ids(
                    example
                )
            )


            is_answerable = bool(
                gold_docs
            )


            decision = (
                router.route(
                    question
                )
            )


            query_type = (
                decision[
                    "query_type"
                ]
            )


            attempt_0 = run_attempt(
                retriever=retriever,
                context_builder=
                    context_builder,
                evidence_grader=
                    evidence_grader,
                original_query=
                    question,
                retrieval_query=
                    question,
                query_type=
                    query_type,
                gold_docs=
                    gold_docs,
            )


            attempts = [
                attempt_0
            ]


            final_attempt = (
                attempt_0
            )


            rewritten = False


            if (
                not attempt_0[
                    "evidence_sufficient"
                ]
                and
                MAX_RETRIES > 0
            ):

                rewritten_query = (
                    rewriter.rewrite(
                        query=question,
                        query_type=query_type,
                        attempt=1,
                    )
                )


                attempt_1 = run_attempt(
                    retriever=retriever,
                    context_builder=
                        context_builder,
                    evidence_grader=
                        evidence_grader,
                    original_query=
                        question,
                    retrieval_query=
                        rewritten_query,
                    query_type=
                        query_type,
                    gold_docs=
                        gold_docs,
                )


                attempts.append(
                    attempt_1
                )

                final_attempt = (
                    attempt_1
                )

                rewritten = True


            records.append(
                {
                    "id":
                        example[
                            "id"
                        ],

                    "question":
                        question,

                    "gold_question_type":
                        example.get(
                            "question_type"
                        ),

                    "predicted_query_type":
                        query_type,

                    "is_answerable":
                        is_answerable,

                    "gold_document_ids":
                        gold_docs,

                    "rewritten":
                        rewritten,

                    "attempts":
                        attempts,

                    "final":
                        final_attempt,
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


    return records


# ============================================================
# Summary
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

    return statistics.mean(
        values
    )


def build_summary(
    records,
):

    answerable = [
        record
        for record in records
        if record[
            "is_answerable"
        ]
    ]


    null_examples = [
        record
        for record in records
        if not record[
            "is_answerable"
        ]
    ]


    # ========================================================
    # Final evidence decisions
    # ========================================================

    answerable_sufficient = [
        record
        for record in answerable
        if record[
            "final"
        ][
            "evidence_sufficient"
        ]
    ]


    null_sufficient = [
        record
        for record in null_examples
        if record[
            "final"
        ][
            "evidence_sufficient"
        ]
    ]


    # ========================================================
    # Gold-context proxy analysis
    # ========================================================

    complete_context = [
        record
        for record in answerable
        if (
            record[
                "final"
            ][
                "context_gold_recall"
            ]
            ==
            1.0
        )
    ]


    complete_context_rejected = [
        record
        for record in complete_context
        if not record[
            "final"
        ][
            "evidence_sufficient"
        ]
    ]


    high_context = [
        record
        for record in answerable
        if (
            record[
                "final"
            ][
                "context_gold_recall"
            ]
            is not None
            and
            record[
                "final"
            ][
                "context_gold_recall"
            ]
            >=
            0.75
        )
    ]


    high_context_rejected = [
        record
        for record in high_context
        if not record[
            "final"
        ][
            "evidence_sufficient"
        ]
    ]


    low_context = [
        record
        for record in answerable
        if (
            record[
                "final"
            ][
                "context_gold_recall"
            ]
            is not None
            and
            record[
                "final"
            ][
                "context_gold_recall"
            ]
            <=
            0.25
        )
    ]


    low_context_accepted = [
        record
        for record in low_context
        if record[
            "final"
        ][
            "evidence_sufficient"
        ]
    ]


    # ========================================================
    # Rewrite behavior
    # ========================================================

    rewritten = [
        record
        for record in records
        if record[
            "rewritten"
        ]
    ]


    rewrite_rescued = []

    rewrite_harmed_recall = []

    rewrite_improved_recall = []


    for record in rewritten:

        first = record[
            "attempts"
        ][0]

        second = record[
            "attempts"
        ][1]


        if (
            not first[
                "evidence_sufficient"
            ]
            and
            second[
                "evidence_sufficient"
            ]
        ):

            rewrite_rescued.append(
                record
            )


        before = first[
            "context_gold_recall"
        ]

        after = second[
            "context_gold_recall"
        ]


        if (
            before is not None
            and
            after is not None
        ):

            if after > before:

                rewrite_improved_recall.append(
                    record
                )

            elif after < before:

                rewrite_harmed_recall.append(
                    record
                )


    # ========================================================
    # Low term coverage rejection
    # ========================================================

    low_term_gate_rejections = [
        record
        for record in answerable
        if (
            not record[
                "final"
            ][
                "evidence_sufficient"
            ]
            and
            record[
                "final"
            ][
                "low_term_coverage"
            ]
            is not None
        )
    ]


    by_type = {}


    grouped = defaultdict(
        list
    )


    for record in records:

        grouped[
            record[
                "predicted_query_type"
            ]
        ].append(
            record
        )


    for query_type, items in (
        sorted(
            grouped.items()
        )
    ):

        type_answerable = [
            item
            for item in items
            if item[
                "is_answerable"
            ]
        ]


        type_null = [
            item
            for item in items
            if not item[
                "is_answerable"
            ]
        ]


        by_type[
            query_type
        ] = {
            "count":
                len(
                    items
                ),

            "answerable_count":
                len(
                    type_answerable
                ),

            "answerable_sufficient_rate":
                (
                    sum(
                        1
                        for item
                        in type_answerable
                        if item[
                            "final"
                        ][
                            "evidence_sufficient"
                        ]
                    )
                    /
                    len(
                        type_answerable
                    )

                    if type_answerable
                    else None
                ),

            "null_count":
                len(
                    type_null
                ),

            "null_sufficient_rate":
                (
                    sum(
                        1
                        for item
                        in type_null
                        if item[
                            "final"
                        ][
                            "evidence_sufficient"
                        ]
                    )
                    /
                    len(
                        type_null
                    )

                    if type_null
                    else None
                ),
        }


    return {
        "total":
            len(
                records
            ),

        "answerable":
            len(
                answerable
            ),

        "null":
            len(
                null_examples
            ),


        # ----------------------------------------------------
        # Main evidence-gate behavior
        # ----------------------------------------------------

        "answerable_sufficient_rate":
            (
                len(
                    answerable_sufficient
                )
                /
                len(
                    answerable
                )

                if answerable
                else None
            ),

        "null_sufficient_rate":
            (
                len(
                    null_sufficient
                )
                /
                len(
                    null_examples
                )

                if null_examples
                else None
            ),

        "null_correct_rejection_rate":
            (
                1.0
                -
                (
                    len(
                        null_sufficient
                    )
                    /
                    len(
                        null_examples
                    )
                )

                if null_examples
                else None
            ),


        # ----------------------------------------------------
        # Dataset-evidence proxy diagnostics
        # ----------------------------------------------------

        "mean_final_retrieved_gold_recall":
            safe_mean(
                record[
                    "final"
                ][
                    "retrieved_gold_recall"
                ]
                for record
                in answerable
            ),

        "mean_final_context_gold_recall":
            safe_mean(
                record[
                    "final"
                ][
                    "context_gold_recall"
                ]
                for record
                in answerable
            ),

        "complete_context_count":
            len(
                complete_context
            ),

        "complete_context_rejected_count":
            len(
                complete_context_rejected
            ),

        "complete_context_rejection_rate":
            (
                len(
                    complete_context_rejected
                )
                /
                len(
                    complete_context
                )

                if complete_context
                else None
            ),

        "high_context_ge_075_count":
            len(
                high_context
            ),

        "high_context_ge_075_rejected_count":
            len(
                high_context_rejected
            ),

        "high_context_ge_075_rejection_rate":
            (
                len(
                    high_context_rejected
                )
                /
                len(
                    high_context
                )

                if high_context
                else None
            ),

        "low_context_le_025_count":
            len(
                low_context
            ),

        "low_context_le_025_accepted_count":
            len(
                low_context_accepted
            ),

        "low_context_le_025_acceptance_rate":
            (
                len(
                    low_context_accepted
                )
                /
                len(
                    low_context
                )

                if low_context
                else None
            ),


        # ----------------------------------------------------
        # Term-coverage hard gate
        # ----------------------------------------------------

        "low_term_coverage_rejection_count":
            len(
                low_term_gate_rejections
            ),

        "mean_rejected_term_coverage":
            safe_mean(

                record[
                    "final"
                ][
                    "low_term_coverage"
                ][
                    "observed"
                ]

                for record
                in low_term_gate_rejections
            ),


        # ----------------------------------------------------
        # Rewrite
        # ----------------------------------------------------

        "rewrite_count":
            len(
                rewritten
            ),

        "rewrite_rate":
            (
                len(
                    rewritten
                )
                /
                len(
                    records
                )

                if records
                else None
            ),

        "rewrite_rescue_count":
            len(
                rewrite_rescued
            ),

        "rewrite_improved_context_recall_count":
            len(
                rewrite_improved_recall
            ),

        "rewrite_harmed_context_recall_count":
            len(
                rewrite_harmed_recall
            ),


        "by_predicted_query_type":
            by_type,
    }


# ============================================================
# Failure samples
# ============================================================

def print_failure_samples(
    records,
):

    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "COMPLETE CONTEXT BUT REJECTED"
    )

    print(
        "=" * 100
    )


    failures = [
        record
        for record in records
        if (
            record[
                "is_answerable"
            ]
            and
            record[
                "final"
            ][
                "context_gold_recall"
            ]
            ==
            1.0
            and
            not record[
                "final"
            ][
                "evidence_sufficient"
            ]
        )
    ]


    for record in failures[:15]:

        final = record[
            "final"
        ]


        print(
            "\n",
            record[
                "id"
            ],
            record[
                "predicted_query_type"
            ],
        )

        print(
            "Score:",
            final[
                "evidence_score"
            ]
        )

        print(
            "Term gate:",
            final[
                "low_term_coverage"
            ]
        )

        print(
            "Reasons:",
            final[
                "evidence_reasons"
            ]
        )


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "LOW CONTEXT RECALL BUT ACCEPTED"
    )

    print(
        "=" * 100
    )


    suspicious = [
        record
        for record in records
        if (
            record[
                "is_answerable"
            ]
            and
            record[
                "final"
            ][
                "context_gold_recall"
            ]
            is not None
            and
            record[
                "final"
            ][
                "context_gold_recall"
            ]
            <=
            0.25
            and
            record[
                "final"
            ][
                "evidence_sufficient"
            ]
        )
    ]


    for record in suspicious[:15]:

        final = record[
            "final"
        ]


        print(
            "\n",
            record[
                "id"
            ],
            record[
                "predicted_query_type"
            ],
        )

        print(
            "Context recall:",
            final[
                "context_gold_recall"
            ]
        )

        print(
            "Score:",
            final[
                "evidence_score"
            ]
        )

        print(
            "Reasons:",
            final[
                "evidence_reasons"
            ]
        )


# ============================================================
# Main
# ============================================================

def main():

    with open(
        FROZEN_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        examples = json.load(
            file
        )


    print(
        "Frozen examples:",
        len(
            examples
        )
    )


    records = run_benchmark(
        examples
    )


    summary = (
        build_summary(
            records
        )
    )


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "EVIDENCE GATE V2 — FROZEN 500"
    )

    print(
        "=" * 100
    )


    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


    print_failure_samples(
        records
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
                    "evidence_gate_v2_500",

                "retrieval":
                    {
                        "collection":
                            COLLECTION_NAME,

                        "bm25_corpus":
                            BM25_CORPUS_PATH,

                        "top_k":
                            TOP_K,
                    },

                "summary":
                    summary,

                "records":
                    records,
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