import json

from pathlib import Path

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)

from adaptive_agentic_rag.orchestration.explicit_source_coverage import (
    ExplicitSourceCoverageGuard,
)

from adaptive_agentic_rag.retrieval.mmr import (
    mmr_select,
)


FROZEN_DATASET_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)


RETRY_DIAGNOSTIC_PATH = Path(
    "evaluation/results/"
    "adaptive_retry_frozen500_diagnostic.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "source_targeted_retry_ablation.json"
)


FINAL_TOP_K = 10

BASE_CANDIDATE_K = 40

SOURCE_CANDIDATE_K_PER_SOURCE = 20

RERANK_CANDIDATE_K = 20


# ============================================================
# Generic JSON
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
# Frozen examples
# ============================================================

def dataset_examples(
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
        "Unsupported frozen-eval dataset structure."
    )


# ============================================================
# Retry records
# ============================================================

def retry_records(
    payload,
):

    records = (
        payload.get(
            "retry_candidates"
        )
    )


    if isinstance(
        records,
        list,
    ):

        return records


    records = (
        payload.get(
            "records"
        )
    )


    if isinstance(
        records,
        list,
    ):

        return [
            record

            for record
            in records

            if (
                record.get(
                    "initial_route"
                )
                ==
                "rewrite"
                or
                record.get(
                    "rewrite_attempted",
                    False,
                )
            )
        ]


    raise ValueError(
        "Cannot locate retry candidates "
        "inside retry diagnostic."
    )


# ============================================================
# Stable normalized source identity
# ============================================================

def normalize_source(
    source: str,
) -> str:

    return (
        ExplicitSourceCoverageGuard
        ._normalize(
            source
        )
    )


def source_aliases(
    source: str,
) -> set[str]:

    return (
        ExplicitSourceCoverageGuard
        ._source_aliases(
            source
        )
    )


# ============================================================
# Result metadata helpers
# ============================================================

def result_source(
    result: dict,
) -> str:

    metadata = (
        result.get(
            "metadata",
            {}
        )
        or {}
    )


    return (
        result.get(
            "source"
        )
        or
        metadata.get(
            "source"
        )
        or
        ""
    )


def result_document_id(
    result: dict,
) -> str | None:

    return (
        result.get(
            "document_id"
        )
    )


# ============================================================
# Context helpers
# ============================================================

def context_document_ids(
    context,
) -> list[str]:

    if context is None:

        return []


    items = (
        getattr(
            context,
            "items",
            [],
        )
        or []
    )


    output = []


    for item in items:

        if isinstance(
            item,
            dict,
        ):

            document_id = (
                item.get(
                    "document_id"
                )
            )


        else:

            document_id = (
                getattr(
                    item,
                    "document_id",
                    None,
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


def context_sources(
    context,
) -> list[str]:

    if context is None:

        return []


    items = (
        getattr(
            context,
            "items",
            [],
        )
        or []
    )


    output = []


    for item in items:

        if isinstance(
            item,
            dict,
        ):

            source = (
                item.get(
                    "source"
                )
                or
                (
                    item.get(
                        "metadata",
                        {}
                    )
                    or {}
                ).get(
                    "source"
                )
            )


        else:

            source = (
                getattr(
                    item,
                    "source",
                    None,
                )
            )


        source = (
            source
            or ""
        ).strip()


        if (
            source
            and
            source
            not in output
        ):

            output.append(
                source
            )


    return output


# ============================================================
# Dataset evidence
# ============================================================

def gold_document_ids(
    example,
) -> list[str]:

    values = (
        example.get(
            "evidence_document_ids"
        )
        or
        example.get(
            "evidence_ids"
        )
        or
        []
    )


    return [
        str(
            value
        )

        for value
        in values
    ]


def document_recall(
    retrieved_ids,
    gold_ids,
):

    gold = set(
        gold_ids
    )


    if not gold:

        return None


    retrieved = set(
        retrieved_ids
    )


    return (
        len(
            gold
            &
            retrieved
        )
        /
        len(
            gold
        )
    )


# ============================================================
# Source matching
# ============================================================

def source_matches(
    actual_source: str,
    required_source: str,
) -> bool:

    actual_aliases = (
        source_aliases(
            actual_source
        )
    )


    required_normalized = (
        normalize_source(
            required_source
        )
    )


    return (
        required_normalized
        in
        actual_aliases
    )


# ============================================================
# Source-targeted BM25
#
# IMPORTANT:
#
# We do NOT rebuild BM25.
#
# We use the production BM25 index and restrict ranking to
# documents whose metadata.source matches the missing source.
# ============================================================

def source_targeted_bm25_search(
    *,
    bm25_retriever,
    query: str,
    missing_sources: list[str],
    top_k_per_source: int,
) -> list[dict]:

    query_tokens = (
        bm25_retriever
        ._tokenize(
            query
        )
    )


    if not query_tokens:

        return []


    scores = (
        bm25_retriever
        .bm25
        .get_scores(
            query_tokens
        )
    )


    output = []


    seen_ids = set()


    for missing_source in (
        missing_sources
    ):

        matching = []


        for (
            index,
            score,
        ) in enumerate(
            scores
        ):

            document = (
                bm25_retriever
                .documents[
                    index
                ]
            )


            metadata = (
                document.get(
                    "metadata",
                    {}
                )
                or {}
            )


            source = (
                metadata.get(
                    "source",
                    ""
                )
                or ""
            )


            if not (
                source_matches(
                    source,
                    missing_source,
                )
            ):

                continue


            matching.append(
                (
                    index,
                    float(
                        score
                    ),
                )
            )


        matching.sort(
            key=lambda item: (
                -item[
                    1
                ],
                item[
                    0
                ],
            )
        )


        for (
            index,
            score,
        ) in matching[
            :top_k_per_source
        ]:

            document = (
                bm25_retriever
                .documents[
                    index
                ]
            )


            document_id = (
                document[
                    "id"
                ]
            )


            if document_id in seen_ids:

                continue


            seen_ids.add(
                document_id
            )


            output.append(
                {
                    "id":
                        document[
                            "id"
                        ],

                    "document_id":
                        document[
                            "document_id"
                        ],

                    "text":
                        document[
                            "text"
                        ],

                    "metadata":
                        document.get(
                            "metadata",
                            {}
                        ),

                    "score":
                        score,

                    # Diagnostic provenance only.
                    "source_targeted":
                        True,

                    "source_target":
                        missing_source,
                }
            )


    return output


# ============================================================
# Candidate merging
# ============================================================

def merge_candidates(
    *candidate_lists,
) -> list[dict]:

    output = []

    seen = set()


    for candidates in (
        candidate_lists
    ):

        for candidate in (
            candidates
        ):

            key = (
                candidate.get(
                    "id"
                )
                or
                (
                    candidate.get(
                        "document_id"
                    ),
                    candidate.get(
                        "text"
                    ),
                )
            )


            if key in seen:

                # If it already exists from normal retrieval,
                # preserve diagnostic source-target flag.

                for existing in output:

                    existing_key = (
                        existing.get(
                            "id"
                        )
                        or
                        (
                            existing.get(
                                "document_id"
                            ),
                            existing.get(
                                "text"
                            ),
                        )
                    )


                    if (
                        existing_key
                        ==
                        key
                    ):

                        if candidate.get(
                            "source_targeted"
                        ):

                            existing[
                                "source_targeted"
                            ] = True


                            existing[
                                "source_target"
                            ] = (
                                candidate.get(
                                    "source_target"
                                )
                            )


                        break


                continue


            seen.add(
                key
            )


            output.append(
                dict(
                    candidate
                )
            )


    return output


# ============================================================
# Production-equivalent finalization
#
# candidate generation differs.
#
# Reranking + vector completion + MMR remain the same.
# ============================================================

def finalize_candidates(
    *,
    reranked_retriever,
    query: str,
    candidates: list[dict],
    top_k: int,
) -> list[dict]:

    if not candidates:

        return []


    reranked = (
        reranked_retriever
        .reranker
        .rerank(
            query,
            candidates,
            top_k=max(
                RERANK_CANDIDATE_K,
                top_k * 2,
            ),
        )
    )


    if not reranked:

        return []


    valid_documents = []

    document_embeddings = []


    for item in reranked:

        vector = (
            item.get(
                "vector"
            )
        )


        if vector is None:

            vector = (
                reranked_retriever
                .hybrid
                .dense
                .embedder
                .encode_documents(
                    [
                        item[
                            "text"
                        ]
                    ]
                )[
                    0
                ]
            )


            item[
                "vector"
            ] = vector


        valid_documents.append(
            item
        )


        document_embeddings.append(
            vector
        )


    query_embedding = (
        reranked_retriever
        .hybrid
        .dense
        .embedder
        .encode_queries(
            [
                query
            ]
        )[
            0
        ]
    )


    selected = (
        mmr_select(
            query_embedding=
                query_embedding,

            document_embeddings=
                document_embeddings,

            documents=
                valid_documents,

            top_k=
                top_k,

            lambda_param=
                reranked_retriever
                .mmr_lambda,
        )
    )


    for item in selected:

        rerank_score = (
            item.get(
                "rerank_score"
            )
        )


        if rerank_score is not None:

            item[
                "score"
            ] = (
                rerank_score
            )


    return selected


# ============================================================
# Result summary helpers
# ============================================================

def result_document_ids(
    results,
):

    output = []


    for result in results:

        document_id = (
            result_document_id(
                result
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


def result_sources(
    results,
):

    output = []


    for result in results:

        source = (
            result_source(
                result
            )
        )


        if (
            source
            and
            source
            not in output
        ):

            output.append(
                source
            )


    return output


def contains_missing_source(
    sources,
    missing_sources,
):

    for missing_source in (
        missing_sources
    ):

        if not any(
            source_matches(
                actual_source,
                missing_source,
            )

            for actual_source
            in sources
        ):

            return False


    return bool(
        missing_sources
    )


# ============================================================
# One ablation case
# ============================================================

def evaluate_case(
    *,
    nodes,
    example,
    retry_record,
):

    question = (
        example.get(
            "question"
        )
        or
        retry_record.get(
            "question"
        )
        or
        ""
    )


    query_decision = (
        nodes.router.route(
            question
        )
    )


    query_type = (
        query_decision[
            "query_type"
        ]
    )


    required_sources = list(
        retry_record.get(
            "initial_required_sources",
            retry_record.get(
                "required_sources",
                [],
            ),
        )
        or []
    )


    covered_sources = list(
        retry_record.get(
            "initial_covered_sources",
            retry_record.get(
                "covered_sources",
                [],
            ),
        )
        or []
    )


    missing_sources = list(
        retry_record.get(
            "initial_missing_sources",
            retry_record.get(
                "missing_sources",
                [],
            ),
        )
        or []
    )


    rewritten_query = (
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


    reranked_retriever = (
        nodes.retriever
        .reranked
    )


    # ========================================================
    # BASELINE RETRY
    #
    # Same candidate generation used by production heavy path:
    #
    # MultiQuery -> Hybrid
    # ========================================================

    baseline_candidates = (
        reranked_retriever
        .multi_query
        .search(
            rewritten_query,
            top_k=
                BASE_CANDIDATE_K,
        )
    )


    baseline_results = (
        finalize_candidates(
            reranked_retriever=
                reranked_retriever,

            query=
                rewritten_query,

            candidates=
                baseline_candidates,

            top_k=
                FINAL_TOP_K,
        )
    )


    # ========================================================
    # SOURCE-TARGETED INJECTION
    # ========================================================

    targeted_candidates = (
        source_targeted_bm25_search(
            bm25_retriever=(
                reranked_retriever
                .hybrid
                .bm25
            ),

            query=
                rewritten_query,

            missing_sources=
                missing_sources,

            top_k_per_source=
                SOURCE_CANDIDATE_K_PER_SOURCE,
        )
    )


    injected_candidate_pool = (
        merge_candidates(
            baseline_candidates,
            targeted_candidates,
        )
    )


    targeted_results = (
        finalize_candidates(
            reranked_retriever=
                reranked_retriever,

            query=
                rewritten_query,

            candidates=
                injected_candidate_pool,

            top_k=
                FINAL_TOP_K,
        )
    )


    # ========================================================
    # Build production contexts
    # ========================================================

    baseline_context = (
        nodes.context_builder.build(
            baseline_results,
            query=
                question,
        )
    )


    targeted_context = (
        nodes.context_builder.build(
            targeted_results,
            query=
                question,
        )
    )


    # ========================================================
    # Production evidence gate
    # ========================================================

    baseline_grade = (
        nodes.grade_evidence(
            {
                "original_query":
                    question,

                "query_type":
                    query_type,

                "context":
                    baseline_context,
            }
        )
    )


    targeted_grade = (
        nodes.grade_evidence(
            {
                "original_query":
                    question,

                "query_type":
                    query_type,

                "context":
                    targeted_context,
            }
        )
    )


    gold_ids = (
        gold_document_ids(
            example
        )
    )


    baseline_result_ids = (
        result_document_ids(
            baseline_results
        )
    )


    targeted_result_ids = (
        result_document_ids(
            targeted_results
        )
    )


    baseline_context_ids = (
        context_document_ids(
            baseline_context
        )
    )


    targeted_context_ids = (
        context_document_ids(
            targeted_context
        )
    )


    baseline_sources = (
        result_sources(
            baseline_results
        )
    )


    targeted_sources = (
        result_sources(
            targeted_results
        )
    )


    baseline_context_source_list = (
        context_sources(
            baseline_context
        )
    )


    targeted_context_source_list = (
        context_sources(
            targeted_context
        )
    )


    return {
        "id":
            example.get(
                "id"
            ),

        "dataset_question_type":
            example.get(
                "question_type"
            ),

        "production_query_type":
            query_type,

        "question":
            question,

        "required_sources":
            required_sources,

        "covered_sources":
            covered_sources,

        "missing_sources":
            missing_sources,

        "rewritten_query":
            rewritten_query,

        # ----------------------------------------------------
        # Candidate diagnostics
        # ----------------------------------------------------

        "baseline_candidate_count":
            len(
                baseline_candidates
            ),

        "targeted_candidate_count":
            len(
                targeted_candidates
            ),

        "injected_candidate_count":
            len(
                injected_candidate_pool
            ),

        "targeted_candidate_sources":
            result_sources(
                targeted_candidates
            ),

        # ----------------------------------------------------
        # Retrieval recall
        # ----------------------------------------------------

        "baseline_retrieval_gold_recall":
            document_recall(
                baseline_result_ids,
                gold_ids,
            ),

        "targeted_retrieval_gold_recall":
            document_recall(
                targeted_result_ids,
                gold_ids,
            ),

        # ----------------------------------------------------
        # Context recall
        # ----------------------------------------------------

        "baseline_context_gold_recall":
            document_recall(
                baseline_context_ids,
                gold_ids,
            ),

        "targeted_context_gold_recall":
            document_recall(
                targeted_context_ids,
                gold_ids,
            ),

        # ----------------------------------------------------
        # Source presence
        # ----------------------------------------------------

        "baseline_missing_source_in_results":
            contains_missing_source(
                baseline_sources,
                missing_sources,
            ),

        "targeted_missing_source_in_results":
            contains_missing_source(
                targeted_sources,
                missing_sources,
            ),

        "baseline_missing_source_in_context":
            contains_missing_source(
                baseline_context_source_list,
                missing_sources,
            ),

        "targeted_missing_source_in_context":
            contains_missing_source(
                targeted_context_source_list,
                missing_sources,
            ),

        "baseline_result_sources":
            baseline_sources,

        "targeted_result_sources":
            targeted_sources,

        "baseline_context_sources":
            baseline_context_source_list,

        "targeted_context_sources":
            targeted_context_source_list,

        # ----------------------------------------------------
        # Evidence gate
        # ----------------------------------------------------

        "baseline_evidence_sufficient":
            baseline_grade[
                "evidence_sufficient"
            ],

        "targeted_evidence_sufficient":
            targeted_grade[
                "evidence_sufficient"
            ],

        "baseline_evidence_score":
            baseline_grade[
                "evidence_score"
            ],

        "targeted_evidence_score":
            targeted_grade[
                "evidence_score"
            ],
    }


# ============================================================
# Summary
# ============================================================

def recall_delta(
    record,
    *,
    baseline_key,
    targeted_key,
):

    baseline = (
        record.get(
            baseline_key
        )
    )


    targeted = (
        record.get(
            targeted_key
        )
    )


    if (
        baseline is None
        or
        targeted is None
    ):

        return None


    return (
        targeted
        -
        baseline
    )


def summarize(
    records,
):

    answerable = [
        record

        for record
        in records

        if (
            record[
                "dataset_question_type"
            ]
            !=
            "null_query"
        )
    ]


    null_records = [
        record

        for record
        in records

        if (
            record[
                "dataset_question_type"
            ]
            ==
            "null_query"
        )
    ]


    retrieval_improved = []

    retrieval_harmed = []

    context_improved = []

    context_harmed = []


    for record in (
        answerable
    ):

        retrieval_delta = (
            recall_delta(
                record,
                baseline_key=(
                    "baseline_retrieval_gold_recall"
                ),
                targeted_key=(
                    "targeted_retrieval_gold_recall"
                ),
            )
        )


        context_delta = (
            recall_delta(
                record,
                baseline_key=(
                    "baseline_context_gold_recall"
                ),
                targeted_key=(
                    "targeted_context_gold_recall"
                ),
            )
        )


        if (
            retrieval_delta
            is not None
        ):

            if retrieval_delta > 0:

                retrieval_improved.append(
                    record[
                        "id"
                    ]
                )


            elif retrieval_delta < 0:

                retrieval_harmed.append(
                    record[
                        "id"
                    ]
                )


        if (
            context_delta
            is not None
        ):

            if context_delta > 0:

                context_improved.append(
                    record[
                        "id"
                    ]
                )


            elif context_delta < 0:

                context_harmed.append(
                    record[
                        "id"
                    ]
                )


    baseline_answerable_accepts = sum(
        bool(
            record[
                "baseline_evidence_sufficient"
            ]
        )

        for record
        in answerable
    )


    targeted_answerable_accepts = sum(
        bool(
            record[
                "targeted_evidence_sufficient"
            ]
        )

        for record
        in answerable
    )


    baseline_null_accepts = sum(
        bool(
            record[
                "baseline_evidence_sufficient"
            ]
        )

        for record
        in null_records
    )


    targeted_null_accepts = sum(
        bool(
            record[
                "targeted_evidence_sufficient"
            ]
        )

        for record
        in null_records
    )


    return {
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

        "retrieval_improved_cases":
            retrieval_improved,

        "retrieval_harmed_cases":
            retrieval_harmed,

        "context_improved_cases":
            context_improved,

        "context_harmed_cases":
            context_harmed,

        "baseline_answerable_evidence_accepts":
            baseline_answerable_accepts,

        "targeted_answerable_evidence_accepts":
            targeted_answerable_accepts,

        "baseline_null_evidence_accepts":
            baseline_null_accepts,

        "targeted_null_evidence_accepts":
            targeted_null_accepts,

        "missing_source_reached_results_baseline":
            sum(
                bool(
                    record[
                        "baseline_missing_source_in_results"
                    ]
                )

                for record
                in records
            ),

        "missing_source_reached_results_targeted":
            sum(
                bool(
                    record[
                        "targeted_missing_source_in_results"
                    ]
                )

                for record
                in records
            ),

        "missing_source_reached_context_baseline":
            sum(
                bool(
                    record[
                        "baseline_missing_source_in_context"
                    ]
                )

                for record
                in records
            ),

        "missing_source_reached_context_targeted":
            sum(
                bool(
                    record[
                        "targeted_missing_source_in_context"
                    ]
                )

                for record
                in records
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    dataset_payload = (
        load_json(
            FROZEN_DATASET_PATH
        )
    )


    diagnostic_payload = (
        load_json(
            RETRY_DIAGNOSTIC_PATH
        )
    )


    examples = (
        dataset_examples(
            dataset_payload
        )
    )


    by_id = {
        example[
            "id"
        ]:
            example

        for example
        in examples
    }


    retries = (
        retry_records(
            diagnostic_payload
        )
    )


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "SOURCE-TARGETED RETRY RETRIEVAL ABLATION"
    )


    print(
        "=" * 100
    )


    print(
        "Cases:",
        len(
            retries
        )
    )


    print(
        "Production files modified: NO"
    )


    nodes = (
        RAGNodes()
    )


    records = []


    try:

        for (
            index,
            retry_record,
        ) in enumerate(
            retries,
            start=1,
        ):

            example_id = (
                retry_record.get(
                    "id"
                )
            )


            example = (
                by_id.get(
                    example_id
                )
            )


            if example is None:

                raise KeyError(
                    (
                        "Retry record missing from "
                        "frozen dataset: "
                        f"{example_id}"
                    )
                )


            result = (
                evaluate_case(
                    nodes=
                        nodes,

                    example=
                        example,

                    retry_record=
                        retry_record,
                )
            )


            records.append(
                result
            )


            print(
                "\n"
                +
                "-" * 100
            )


            print(
                (
                    f"{index}/"
                    f"{len(retries)} "
                    f"| "
                    f"{result['id']} "
                    f"| "
                    f"{result['dataset_question_type']}"
                )
            )


            print(
                "Missing:",
                result[
                    "missing_sources"
                ],
            )


            print(
                "Rewrite:"
            )


            print(
                result[
                    "rewritten_query"
                ]
            )


            print(
                (
                    "Targeted BM25 candidates: "
                    f"{result['targeted_candidate_count']}"
                )
            )


            print(
                (
                    "Missing source in final retrieval: "
                    f"{result['baseline_missing_source_in_results']}"
                    " -> "
                    f"{result['targeted_missing_source_in_results']}"
                )
            )


            print(
                (
                    "Missing source in final context: "
                    f"{result['baseline_missing_source_in_context']}"
                    " -> "
                    f"{result['targeted_missing_source_in_context']}"
                )
            )


            print(
                (
                    "Retrieval gold recall: "
                    f"{result['baseline_retrieval_gold_recall']}"
                    " -> "
                    f"{result['targeted_retrieval_gold_recall']}"
                )
            )


            print(
                (
                    "Context gold recall: "
                    f"{result['baseline_context_gold_recall']}"
                    " -> "
                    f"{result['targeted_context_gold_recall']}"
                )
            )


            print(
                (
                    "Evidence sufficient: "
                    f"{result['baseline_evidence_sufficient']}"
                    " -> "
                    f"{result['targeted_evidence_sufficient']}"
                )
            )


            print(
                (
                    "Evidence score: "
                    f"{result['baseline_evidence_score']:.4f}"
                    " -> "
                    f"{result['targeted_evidence_score']:.4f}"
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


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "configuration": {
                    "final_top_k":
                        FINAL_TOP_K,

                    "base_candidate_k":
                        BASE_CANDIDATE_K,

                    "source_candidate_k_per_source":
                        SOURCE_CANDIDATE_K_PER_SOURCE,

                    "rerank_candidate_k":
                        RERANK_CANDIDATE_K,

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