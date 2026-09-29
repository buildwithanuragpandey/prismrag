import json

from dataclasses import dataclass
from pathlib import Path

from adaptive_agentic_rag.orchestration.explicit_source_coverage import (
    ExplicitSourceCoverageGuard,
)


RESULTS_PATH = Path(
    "evaluation/results/"
    "e2e_smoke_20_results.json"
)


CORPUS_PATH = Path(
    "data/processed/"
    "processed_corpus_v2.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "final_claim_source_coverage_diagnostic.json"
)


# ============================================================
# Minimal context representation
#
# ExplicitSourceCoverageGuard only needs:
#
#     context.items[*].source
#
# Therefore this diagnostic does NOT need to rebuild the
# original retrieval context.
# ============================================================

@dataclass
class DiagnosticContextItem:

    source: str


@dataclass
class DiagnosticContext:

    items: list[DiagnosticContextItem]


# ============================================================
# JSON
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
# E2E records
# ============================================================

def load_records():

    payload = (
        load_json(
            RESULTS_PATH
        )
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

        records = (
            payload.get(
                "records"
            )
        )


        if isinstance(
            records,
            list,
        ):

            return records


    raise ValueError(
        "Unsupported E2E results structure."
    )


# ============================================================
# Corpus structure
# ============================================================

def corpus_records(
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
            "chunks",
            "documents",
            "records",
            "items",
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
        "Unsupported processed corpus structure."
    )


# ============================================================
# Document → source catalog
# ============================================================

def build_document_source_catalog():

    payload = (
        load_json(
            CORPUS_PATH
        )
    )


    records = (
        corpus_records(
            payload
        )
    )


    catalog = {}


    for record in records:

        if not isinstance(
            record,
            dict,
        ):

            continue


        document_id = (
            record.get(
                "document_id"
            )
        )


        metadata = (
            record.get(
                "metadata",
                {}
            )
            or {}
        )


        source = (
            metadata.get(
                "source"
            )
            or
            record.get(
                "source"
            )
            or
            ""
        )


        source = (
            str(
                source
            )
            .strip()
        )


        if (
            not document_id
            or
            not source
        ):

            continue


        document_id = (
            str(
                document_id
            )
        )


        catalog.setdefault(
            document_id,
            []
        )


        if (
            source
            not in
            catalog[
                document_id
            ]
        ):

            catalog[
                document_id
            ].append(
                source
            )


    return catalog


# ============================================================
# Stable source collection
# ============================================================

def cited_sources(
    *,
    record,
    catalog,
):

    output = []

    seen = set()


    for document_id in (
        record.get(
            "cited_document_ids",
            [],
        )
        or []
    ):

        document_id = (
            str(
                document_id
            )
        )


        for source in (
            catalog.get(
                document_id,
                [],
            )
        ):

            key = (
                ExplicitSourceCoverageGuard
                ._normalize(
                    source
                )
            )


            if (
                not key
                or
                key in seen
            ):

                continue


            seen.add(
                key
            )


            output.append(
                source
            )


    return output


# ============================================================
# Final cited context
# ============================================================

def build_final_claim_context(
    sources,
):

    return DiagnosticContext(
        items=[
            DiagnosticContextItem(
                source=
                    source
            )

            for source
            in sources
        ]
    )


# ============================================================
# One record
# ============================================================

def analyze_record(
    *,
    record,
    guard,
    catalog,
):

    sources = (
        cited_sources(
            record=
                record,

            catalog=
                catalog,
        )
    )


    context = (
        build_final_claim_context(
            sources
        )
    )


    result = (
        guard.check(
            query=(
                record.get(
                    "question"
                )
                or ""
            ),

            context=
                context,
        )
    )


    answerable = (
        record.get(
            "question_type"
        )
        !=
        "null_query"
    )


    answered = (
        answerable
        and
        not bool(
            record.get(
                "abstained",
                False,
            )
        )
    )


    runtime_passed = bool(
        record.get(
            "runtime_grader_passed",
            False,
        )
    )


    smoke_correct = (
        record.get(
            "smoke_answer_correct"
        )
    )


    dangerous_wrong_answer = (
        answered
        and
        runtime_passed
        and
        smoke_correct
        is False
    )


    correct_answer = (
        answered
        and
        smoke_correct
        is True
    )


    return {
        "id":
            record.get(
                "id"
            ),

        "question_type":
            record.get(
                "question_type"
            ),

        "question":
            record.get(
                "question"
            ),

        "gold_answer":
            record.get(
                "gold_answer"
            ),

        "direct_answer":
            record.get(
                "direct_answer"
            ),

        "abstained":
            bool(
                record.get(
                    "abstained",
                    False,
                )
            ),

        "runtime_grader_passed":
            runtime_passed,

        "smoke_answer_correct":
            smoke_correct,

        "dangerous_wrong_answer":
            dangerous_wrong_answer,

        "correct_answer":
            correct_answer,

        "cited_document_ids":
            list(
                record.get(
                    "cited_document_ids",
                    [],
                )
                or []
            ),

        "final_cited_sources":
            sources,

        "final_source_coverage_satisfied":
            result.satisfied,

        "required_sources":
            result.required_sources,

        "covered_sources":
            result.covered_sources,

        "missing_sources":
            result.missing_sources,
    }


# ============================================================
# Summary
# ============================================================

def summarize(
    records,
):

    answered = [
        record

        for record
        in records

        if (
            record[
                "question_type"
            ]
            !=
            "null_query"
            and
            not record[
                "abstained"
            ]
        )
    ]


    correct = [
        record

        for record
        in answered

        if record[
            "smoke_answer_correct"
        ]
        is True
    ]


    wrong_runtime_passes = [
        record

        for record
        in answered

        if (
            record[
                "runtime_grader_passed"
            ]
            and
            record[
                "smoke_answer_correct"
            ]
            is False
        )
    ]


    wrong_rejected_by_coverage = [
        record

        for record
        in wrong_runtime_passes

        if not record[
            "final_source_coverage_satisfied"
        ]
    ]


    wrong_not_rejected = [
        record

        for record
        in wrong_runtime_passes

        if record[
            "final_source_coverage_satisfied"
        ]
    ]


    correct_rejected_by_coverage = [
        record

        for record
        in correct

        if not record[
            "final_source_coverage_satisfied"
        ]
    ]


    correct_preserved = [
        record

        for record
        in correct

        if record[
            "final_source_coverage_satisfied"
        ]
    ]


    return {
        "answered_answerable":
            len(
                answered
            ),

        "smoke_correct_answered":
            len(
                correct
            ),

        "wrong_runtime_passes":
            len(
                wrong_runtime_passes
            ),

        # ----------------------------------------------------
        # Diagnostic effectiveness
        # ----------------------------------------------------

        "wrong_runtime_passes_rejected_by_final_source_coverage":
            len(
                wrong_rejected_by_coverage
            ),

        "wrong_runtime_passes_rejected_ids": [
            record[
                "id"
            ]

            for record
            in wrong_rejected_by_coverage
        ],

        "wrong_runtime_passes_not_rejected":
            len(
                wrong_not_rejected
            ),

        "wrong_runtime_passes_not_rejected_ids": [
            record[
                "id"
            ]

            for record
            in wrong_not_rejected
        ],

        # ----------------------------------------------------
        # Safety / collateral damage
        # ----------------------------------------------------

        "correct_answers_preserved":
            len(
                correct_preserved
            ),

        "correct_answers_rejected":
            len(
                correct_rejected_by_coverage
            ),

        "correct_answers_rejected_ids": [
            record[
                "id"
            ]

            for record
            in correct_rejected_by_coverage
        ],

        "wrong_rejection_rate":
            (
                len(
                    wrong_rejected_by_coverage
                )
                /
                len(
                    wrong_runtime_passes
                )

                if wrong_runtime_passes

                else None
            ),

        "correct_preservation_rate":
            (
                len(
                    correct_preserved
                )
                /
                len(
                    correct
                )

                if correct

                else None
            ),
    }


# ============================================================
# Console
# ============================================================

def print_case(
    *,
    index,
    total,
    record,
):

    print(
        "\n"
        +
        "-" * 100
    )


    print(
        (
            f"{index}/{total}"
            f" | {record['id']}"
            f" | {record['question_type']}"
        )
    )


    print(
        "Gold:",
        record[
            "gold_answer"
        ],
    )


    print(
        "Direct:",
        record[
            "direct_answer"
        ],
    )


    print(
        "Smoke correct:",
        record[
            "smoke_answer_correct"
        ],
    )


    print(
        "Final cited sources:",
        record[
            "final_cited_sources"
        ],
    )


    print(
        "Required sources:",
        record[
            "required_sources"
        ],
    )


    print(
        "Covered sources:",
        record[
            "covered_sources"
        ],
    )


    print(
        "Missing sources:",
        record[
            "missing_sources"
        ],
    )


    print(
        (
            "Final source coverage: "
            f"{record['final_source_coverage_satisfied']}"
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    records = (
        load_records()
    )


    catalog = (
        build_document_source_catalog()
    )


    guard = (
        ExplicitSourceCoverageGuard()
    )


    analyzed = [
        analyze_record(
            record=
                record,

            guard=
                guard,

            catalog=
                catalog,
        )

        for record
        in records
    ]


    answered = [
        record

        for record
        in analyzed

        if (
            record[
                "question_type"
            ]
            !=
            "null_query"
            and
            not record[
                "abstained"
            ]
        )
    ]


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "FINAL CLAIM SOURCE COVERAGE DIAGNOSTIC"
    )


    print(
        "=" * 100
    )


    print(
        "Model execution: NONE"
    )


    print(
        "Production files modified: NO"
    )


    print(
        "Answered answerable:",
        len(
            answered
        )
    )


    for (
        index,
        record,
    ) in enumerate(
        answered,
        start=1,
    ):

        print_case(
            index=
                index,

            total=
                len(
                    answered
                ),

            record=
                record,
        )


    summary = (
        summarize(
            analyzed
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
                "input_results":
                    str(
                        RESULTS_PATH
                    ),

                "corpus":
                    str(
                        CORPUS_PATH
                    ),

                "production_files_modified":
                    False,

                "summary":
                    summary,

                "records":
                    analyzed,
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