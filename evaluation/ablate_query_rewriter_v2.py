import json
import re

from pathlib import Path

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)


DATASET_PATH = Path(
    "evaluation/datasets/"
    "frozen_eval_500.json"
)


DIAGNOSTIC_PATH = Path(
    "evaluation/results/"
    "adaptive_retry_frozen500_diagnostic.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "query_rewriter_v2_ablation.json"
)


TOP_K = 10


# ============================================================
# Legacy QueryRewriter
#
# Exact behavioral family used before V2 source-targeted
# rewriting.
# ============================================================

class LegacyQueryRewriter:

    @staticmethod
    def _normalize(
        query: str,
    ) -> str:

        return " ".join(
            (
                query
                or ""
            )
            .strip()
            .split()
        )


    def _rewrite_comparison(
        self,
        query: str,
    ) -> str:

        text = (
            self._normalize(
                query
            )
        )


        text = re.sub(
            r"^\s*compare\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )


        text = re.sub(
            (
                r"^\s*comparison\s+"
                r"(?:of|between)\s+"
            ),
            "",
            text,
            flags=re.IGNORECASE,
        )


        if not re.search(
            r"\bcomparison\b",
            text,
            flags=re.IGNORECASE,
        ):

            text = (
                text.rstrip(
                    " ?."
                )
                +
                " comparison"
            )


        return self._normalize(
            text
        )


    def _rewrite_reasoning(
        self,
        query: str,
    ) -> str:

        text = (
            self._normalize(
                query
            )
        )


        if re.match(
            r"^\s*why\b",
            text,
            flags=re.IGNORECASE,
        ):

            text = re.sub(
                r"^\s*why\b",
                "",
                text,
                count=1,
                flags=re.IGNORECASE,
            )


            return self._normalize(
                f"{text} reasons explanation"
            )


        if re.match(
            r"^\s*how\b",
            text,
            flags=re.IGNORECASE,
        ):

            return self._normalize(
                f"{text} explanation"
            )


        return self._normalize(
            f"{text} supporting evidence"
        )


    def _rewrite_complex(
        self,
        query: str,
    ) -> str:

        text = (
            self._normalize(
                query
            )
        )


        if re.search(
            (
                r"\bcompare\b|"
                r"\bcomparison\b|"
                r"\bversus\b|"
                r"\bvs\.?\b"
            ),
            text,
            flags=re.IGNORECASE,
        ):

            return self._rewrite_comparison(
                text
            )


        return self._normalize(
            f"{text} relevant facts evidence"
        )


    def rewrite(
        self,
        query: str,
        query_type: str,
        attempt: int = 1,
    ) -> str:

        text = (
            self._normalize(
                query
            )
        )


        if not text:

            return text


        query_type = (
            query_type
            or ""
        ).strip().lower()


        if re.search(
            (
                r"^\s*compare\b|"
                r"\bcomparison\b|"
                r"\bversus\b|"
                r"\bvs\.?\b"
            ),
            text,
            flags=re.IGNORECASE,
        ):

            rewritten = (
                self._rewrite_comparison(
                    text
                )
            )


        elif (
            query_type
            ==
            "complex"
        ):

            rewritten = (
                self._rewrite_complex(
                    text
                )
            )


        elif (
            query_type
            ==
            "multihop"
        ):

            rewritten = (
                self._rewrite_reasoning(
                    text
                )
            )


        else:

            rewritten = (
                text
            )


        if (
            attempt
            >
            1
            and
            rewritten
            ==
            text
        ):

            rewritten = (
                self._normalize(
                    (
                        f"{rewritten} "
                        "relevant information"
                    )
                )
            )


        return rewritten


# ============================================================
# Load JSON
# ============================================================

def load_json(
    path: Path,
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
# Dataset structure
# ============================================================

def examples_from_payload(
    payload,
):

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

            value = (
                payload.get(
                    key
                )
            )


            if isinstance(
                value,
                list,
            ):

                return value


    raise ValueError(
        "Unsupported frozen dataset structure."
    )


# ============================================================
# Result/context document IDs
# ============================================================

def result_document_ids(
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


def context_document_ids(
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
# Gold recall
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
# Execute one rewrite variant
# ============================================================

def run_variant(
    *,
    nodes,
    original_query,
    rewritten_query,
    query_type,
    missing_sources,
    gold_ids,
):

    # ========================================================
    # IMPORTANT:
    #
    # Source-targeted retrieval is FIXED for both variants.
    #
    # Only rewritten_query changes.
    # ========================================================

    retrieval_output = (
        nodes.retriever.search(
            rewritten_query,
            top_k=
                TOP_K,

            target_sources=
                missing_sources,
        )
    )


    results = (
        retrieval_output[
            "results"
        ]
    )


    retrieved_ids = (
        result_document_ids(
            results
        )
    )


    context = (
        nodes.context_builder.build(
            results,
            query=
                original_query,
        )
    )


    context_ids = (
        context_document_ids(
            context
        )
    )


    evidence = (
        nodes.grade_evidence(
            {
                "original_query":
                    original_query,

                "query_type":
                    query_type,

                "context":
                    context,
            }
        )
    )


    return {
        "retrieved_document_ids":
            retrieved_ids,

        "context_document_ids":
            context_ids,

        "retrieval_gold_recall":
            gold_recall(
                retrieved_ids,
                gold_ids,
            ),

        "context_gold_recall":
            gold_recall(
                context_ids,
                gold_ids,
            ),

        "evidence_sufficient":
            bool(
                evidence[
                    "evidence_sufficient"
                ]
            ),

        "evidence_score":
            evidence[
                "evidence_score"
            ],
    }


# ============================================================
# Delta
# ============================================================

def numeric_delta(
    baseline,
    candidate,
):

    if (
        baseline is None
        or
        candidate is None
    ):

        return None


    return (
        candidate
        -
        baseline
    )


# ============================================================
# Main
# ============================================================

def main():

    dataset_payload = (
        load_json(
            DATASET_PATH
        )
    )


    diagnostic_payload = (
        load_json(
            DIAGNOSTIC_PATH
        )
    )


    examples = (
        examples_from_payload(
            dataset_payload
        )
    )


    examples_by_id = {
        example[
            "id"
        ]:
            example

        for example
        in examples
    }


    retry_candidates = list(
        diagnostic_payload.get(
            "retry_candidates",
            [],
        )
        or []
    )


    if not retry_candidates:

        raise ValueError(
            "No retry candidates found in diagnostic."
        )


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "QUERY REWRITER V2 ABLATION"
    )


    print(
        "=" * 100
    )


    print(
        "Retry cases:",
        len(
            retry_candidates
        )
    )


    print(
        "Source-targeted retrieval: FIXED / ENABLED"
    )


    print(
        "Production files modified: NO"
    )


    nodes = (
        RAGNodes()
    )


    legacy_rewriter = (
        LegacyQueryRewriter()
    )


    records = []


    try:

        for (
            index,
            retry_record,
        ) in enumerate(
            retry_candidates,
            start=1,
        ):

            example_id = (
                retry_record[
                    "id"
                ]
            )


            example = (
                examples_by_id[
                    example_id
                ]
            )


            question = (
                example[
                    "question"
                ]
            )


            dataset_type = (
                example[
                    "question_type"
                ]
            )


            null_example = (
                dataset_type
                ==
                "null_query"
            )


            gold_ids = list(
                example.get(
                    "evidence_document_ids",
                    [],
                )
                or []
            )


            missing_sources = list(
                retry_record.get(
                    "initial_missing_sources",
                    [],
                )
                or []
            )


            required_sources = list(
                retry_record.get(
                    "initial_required_sources",
                    [],
                )
                or []
            )


            covered_sources = list(
                retry_record.get(
                    "initial_covered_sources",
                    [],
                )
                or []
            )


            route_decision = (
                nodes.router.route(
                    question
                )
            )


            query_type = (
                route_decision[
                    "query_type"
                ]
            )


            # =================================================
            # Legacy
            # =================================================

            legacy_query = (
                legacy_rewriter.rewrite(
                    query=
                        question,

                    query_type=
                        query_type,

                    attempt=
                        1,
                )
            )


            legacy_result = (
                run_variant(
                    nodes=
                        nodes,

                    original_query=
                        question,

                    rewritten_query=
                        legacy_query,

                    query_type=
                        query_type,

                    missing_sources=
                        missing_sources,

                    gold_ids=
                        gold_ids,
                )
            )


            # =================================================
            # V2
            # =================================================

            v2_query = (
                nodes.query_rewriter.rewrite(
                    query=
                        question,

                    query_type=
                        query_type,

                    attempt=
                        1,

                    required_sources=
                        required_sources,

                    covered_sources=
                        covered_sources,

                    missing_sources=
                        missing_sources,
                )
            )


            v2_result = (
                run_variant(
                    nodes=
                        nodes,

                    original_query=
                        question,

                    rewritten_query=
                        v2_query,

                    query_type=
                        query_type,

                    missing_sources=
                        missing_sources,

                    gold_ids=
                        gold_ids,
                )
            )


            record = {
                "id":
                    example_id,

                "question_type":
                    dataset_type,

                "is_null":
                    null_example,

                "missing_sources":
                    missing_sources,

                "legacy_query":
                    legacy_query,

                "v2_query":
                    v2_query,

                "legacy":
                    legacy_result,

                "v2":
                    v2_result,

                "retrieval_recall_delta_v2_minus_legacy":
                    numeric_delta(
                        legacy_result[
                            "retrieval_gold_recall"
                        ],
                        v2_result[
                            "retrieval_gold_recall"
                        ],
                    ),

                "context_recall_delta_v2_minus_legacy":
                    numeric_delta(
                        legacy_result[
                            "context_gold_recall"
                        ],
                        v2_result[
                            "context_gold_recall"
                        ],
                    ),

                "evidence_score_delta_v2_minus_legacy":
                    numeric_delta(
                        legacy_result[
                            "evidence_score"
                        ],
                        v2_result[
                            "evidence_score"
                        ],
                    ),

                "legacy_evidence_sufficient":
                    legacy_result[
                        "evidence_sufficient"
                    ],

                "v2_evidence_sufficient":
                    v2_result[
                        "evidence_sufficient"
                    ],
            }


            records.append(
                record
            )


            print(
                "\n"
                +
                "-" * 100
            )


            print(
                (
                    f"{index}/"
                    f"{len(retry_candidates)}"
                    f" | {example_id}"
                    f" | {dataset_type}"
                )
            )


            print(
                "Missing:",
                missing_sources,
            )


            print(
                "\nLEGACY:"
            )


            print(
                legacy_query
            )


            print(
                "\nV2:"
            )


            print(
                v2_query
            )


            print(
                (
                    "\nRetrieval recall: "
                    f"{legacy_result['retrieval_gold_recall']}"
                    " -> "
                    f"{v2_result['retrieval_gold_recall']}"
                )
            )


            print(
                (
                    "Context recall: "
                    f"{legacy_result['context_gold_recall']}"
                    " -> "
                    f"{v2_result['context_gold_recall']}"
                )
            )


            print(
                (
                    "Evidence sufficient: "
                    f"{legacy_result['evidence_sufficient']}"
                    " -> "
                    f"{v2_result['evidence_sufficient']}"
                )
            )


            print(
                (
                    "Evidence score: "
                    f"{legacy_result['evidence_score']:.4f}"
                    " -> "
                    f"{v2_result['evidence_score']:.4f}"
                )
            )


    finally:

        nodes.close()


    # ========================================================
    # Summary
    # ========================================================

    answerable = [
        record

        for record
        in records

        if not record[
            "is_null"
        ]
    ]


    null_records = [
        record

        for record
        in records

        if record[
            "is_null"
        ]
    ]


    summary = {
        "total_cases":
            len(
                records
            ),

        "answerable_cases":
            len(
                answerable
            ),

        "null_cases":
            len(
                null_records
            ),

        "legacy_answerable_evidence_accepts":
            sum(
                record[
                    "legacy_evidence_sufficient"
                ]

                for record
                in answerable
            ),

        "v2_answerable_evidence_accepts":
            sum(
                record[
                    "v2_evidence_sufficient"
                ]

                for record
                in answerable
            ),

        "legacy_null_evidence_accepts":
            sum(
                record[
                    "legacy_evidence_sufficient"
                ]

                for record
                in null_records
            ),

        "v2_null_evidence_accepts":
            sum(
                record[
                    "v2_evidence_sufficient"
                ]

                for record
                in null_records
            ),

        "v2_retrieval_improved_cases": [
            record[
                "id"
            ]

            for record
            in answerable

            if (
                record[
                    "retrieval_recall_delta_v2_minus_legacy"
                ]
                is not None
                and
                record[
                    "retrieval_recall_delta_v2_minus_legacy"
                ]
                >
                0
            )
        ],

        "v2_retrieval_harmed_cases": [
            record[
                "id"
            ]

            for record
            in answerable

            if (
                record[
                    "retrieval_recall_delta_v2_minus_legacy"
                ]
                is not None
                and
                record[
                    "retrieval_recall_delta_v2_minus_legacy"
                ]
                <
                0
            )
        ],

        "v2_context_improved_cases": [
            record[
                "id"
            ]

            for record
            in answerable

            if (
                record[
                    "context_recall_delta_v2_minus_legacy"
                ]
                is not None
                and
                record[
                    "context_recall_delta_v2_minus_legacy"
                ]
                >
                0
            )
        ],

        "v2_context_harmed_cases": [
            record[
                "id"
            ]

            for record
            in answerable

            if (
                record[
                    "context_recall_delta_v2_minus_legacy"
                ]
                is not None
                and
                record[
                    "context_recall_delta_v2_minus_legacy"
                ]
                <
                0
            )
        ],
    }


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
                "configuration": {
                    "source_targeted_retrieval":
                        True,

                    "top_k":
                        TOP_K,

                    "production_files_modified":
                        False,
                },

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