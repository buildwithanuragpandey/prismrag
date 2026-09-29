import json
import math
import statistics

from pathlib import Path

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)

from adaptive_agentic_rag.retrieval.query_decomposer import (
    QueryDecomposer,
)


DATASET_PATH = Path(
    "evaluation/datasets/"
    "frozen_e2e_smoke_20.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "semantic_requirement_score_diagnostic.json"
)


# ============================================================
# Basic utilities
# ============================================================

def mean(
    values,
):

    values = [
        float(value)
        for value
        in values
        if value is not None
    ]


    if not values:

        return None


    return (
        sum(values)
        /
        len(values)
    )


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
        len(gold)
    )


# ============================================================
# Requirement extraction
#
# IMPORTANT:
#
# We intentionally reuse the CURRENT QueryDecomposer unchanged.
#
# We already know it is imperfect.
#
# This diagnostic asks only:
#
# "Does semantic scoring help despite the current decomposer?"
#
# No decomposer patch is introduced here.
# ============================================================

class SemanticRequirementAnalyzer:

    def __init__(
        self,
        reranker,
        evidence_grader,
    ):

        self.reranker = (
            reranker
        )


        self.evidence_grader = (
            evidence_grader
        )


        self.decomposer = (
            QueryDecomposer()
        )


    # ========================================================
    # Requirements
    # ========================================================

    def extract_requirements(
        self,
        query: str,
    ) -> list[str]:

        output = (
            self.decomposer.decompose(
                query
            )
        )


        if not output:

            return [
                query
            ]


        # ----------------------------------------------------
        # QueryDecomposer output:
        #
        # original query
        # +
        # facets
        #
        # When facets exist, don't treat the original monster
        # query as another requirement.
        # ----------------------------------------------------

        if len(output) > 1:

            requirements = (
                output[
                    1:
                ]
            )

        else:

            requirements = [
                output[
                    0
                ]
            ]


        unique = []

        seen = set()


        for requirement in requirements:

            requirement = (
                " ".join(
                    requirement.split()
                )
            )


            key = (
                requirement.lower()
            )


            if (
                not requirement
                or
                key in seen
            ):

                continue


            seen.add(
                key
            )


            unique.append(
                requirement
            )


        return (
            unique
            or
            [
                query
            ]
        )


    # ========================================================
    # Context candidates
    #
    # Score chunks instead of concatenating a full document.
    #
    # Reason:
    #
    # BGE reranker truncates query/document pairs to 512
    # tokens. Concatenating an entire document here could hide
    # the useful passage behind truncation.
    # ========================================================

    @staticmethod
    def build_candidates(
        context,
    ) -> list[dict]:

        candidates = []


        for (
            index,
            item,
        ) in enumerate(
            context.items
        ):

            parts = []


            if item.source:

                parts.append(
                    f"Source: {item.source}"
                )


            if item.title:

                parts.append(
                    f"Title: {item.title}"
                )


            parts.append(
                f"Evidence: {item.text}"
            )


            searchable_text = (
                "\n".join(
                    parts
                )
            )


            candidates.append(
                {
                    "id":
                        f"context_{index}",

                    "text":
                        searchable_text,

                    "citation_id":
                        item.citation_id,

                    "chunk_id":
                        item.chunk_id,

                    "document_id":
                        item.document_id,

                    "source":
                        item.source or "",

                    "title":
                        item.title or "",
                }
            )


        return candidates


    # ========================================================
    # Critical-anchor coverage
    #
    # Reuse the exact semantics already implemented in V2.
    # ========================================================

    def critical_coverage(
        self,
        text: str,
        critical_terms: set[str],
    ) -> float:

        if not critical_terms:

            return 1.0


        matched = {

            anchor

            for anchor
            in critical_terms

            if (
                self.evidence_grader
                ._anchor_present(
                    text,
                    anchor,
                )
            )
        }


        return (
            len(matched)
            /
            len(critical_terms)
        )


    # ========================================================
    # Analyze one requirement
    # ========================================================

    def score_requirement(
        self,
        requirement: str,
        candidates: list[dict],
        query_type: str,
        gold_document_ids: list[str],
    ) -> dict:

        critical_terms = (
            self.evidence_grader
            ._critical_terms(
                requirement
            )
        )


        required_critical_coverage = (
            self.evidence_grader
            ._critical_coverage_requirement(
                query_type,
                critical_terms,
            )
        )


        ranked = (
            self.reranker.rerank(
                query=
                    requirement,

                documents=
                    candidates,

                top_k=
                    len(
                        candidates
                    ),
            )
        )


        if not ranked:

            return {
                "requirement":
                    requirement,

                "critical_terms":
                    sorted(
                        critical_terms
                    ),

                "required_critical_coverage":
                    required_critical_coverage,

                "best_score":
                    None,

                "second_score":
                    None,

                "score_margin":
                    None,

                "best_document_id":
                    None,

                "best_chunk_id":
                    None,

                "best_source":
                    None,

                "best_title":
                    None,

                "best_critical_coverage":
                    0.0,

                "best_anchor_ok":
                    False,

                "best_document_is_gold":
                    None,

                "top3":
                    [],
            }


        best = (
            ranked[
                0
            ]
        )


        second_score = None


        if len(ranked) > 1:

            second_score = float(
                ranked[
                    1
                ][
                    "rerank_score"
                ]
            )


        best_score = float(
            best[
                "rerank_score"
            ]
        )


        score_margin = None


        if (
            second_score
            is not None
        ):

            score_margin = (
                best_score
                -
                second_score
            )


        best_critical_coverage = (
            self.critical_coverage(
                text=
                    best[
                        "text"
                    ],

                critical_terms=
                    critical_terms,
            )
        )


        best_anchor_ok = (

            best_critical_coverage

            >=

            required_critical_coverage
        )


        gold = set(
            gold_document_ids
            or []
        )


        best_document_is_gold = None


        if gold:

            best_document_is_gold = (

                best[
                    "document_id"
                ]

                in

                gold
            )


        top3 = []


        for item in (
            ranked[
                :3
            ]
        ):

            top3.append(
                {
                    "document_id":
                        item[
                            "document_id"
                        ],

                    "chunk_id":
                        item[
                            "chunk_id"
                        ],

                    "citation_id":
                        item[
                            "citation_id"
                        ],

                    "source":
                        item[
                            "source"
                        ],

                    "title":
                        item[
                            "title"
                        ],

                    "score":
                        round(
                            float(
                                item[
                                    "rerank_score"
                                ]
                            ),
                            4,
                        ),

                    "is_gold_document": (
                        (
                            item[
                                "document_id"
                            ]
                            in
                            gold
                        )
                        if gold
                        else None
                    ),
                }
            )


        return {
            "requirement":
                requirement,

            "critical_terms":
                sorted(
                    critical_terms
                ),

            "required_critical_coverage":
                round(
                    required_critical_coverage,
                    4,
                ),

            "best_score":
                round(
                    best_score,
                    4,
                ),

            "second_score": (
                round(
                    second_score,
                    4,
                )
                if (
                    second_score
                    is not None
                )
                else None
            ),

            "score_margin": (
                round(
                    score_margin,
                    4,
                )
                if (
                    score_margin
                    is not None
                )
                else None
            ),

            "best_document_id":
                best[
                    "document_id"
                ],

            "best_chunk_id":
                best[
                    "chunk_id"
                ],

            "best_source":
                best[
                    "source"
                ],

            "best_title":
                best[
                    "title"
                ],

            "best_critical_coverage":
                round(
                    best_critical_coverage,
                    4,
                ),

            "best_anchor_ok":
                best_anchor_ok,

            "best_document_is_gold":
                best_document_is_gold,

            "top3":
                top3,
        }


# ============================================================
# Run one frozen example
# ============================================================

def run_case(
    nodes,
    analyzer,
    example,
):

    question = (
        example[
            "question"
        ]
    )


    gold_document_ids = (
        example.get(
            "evidence_document_ids",
            [],
        )
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


    candidates = (
        analyzer.build_candidates(
            context
        )
    )


    requirements = (
        analyzer.extract_requirements(
            question
        )
    )


    requirement_results = []


    for requirement in requirements:

        requirement_results.append(
            analyzer.score_requirement(
                requirement=
                    requirement,

                candidates=
                    candidates,

                query_type=
                    query_type,

                gold_document_ids=
                    gold_document_ids,
            )
        )


    best_scores = [

        item[
            "best_score"
        ]

        for item
        in requirement_results

        if (
            item[
                "best_score"
            ]
            is not None
        )
    ]


    margins = [

        item[
            "score_margin"
        ]

        for item
        in requirement_results

        if (
            item[
                "score_margin"
            ]
            is not None
        )
    ]


    anchor_ok_count = sum(

        1

        for item
        in requirement_results

        if item[
            "best_anchor_ok"
        ]
    )


    gold_best_flags = [

        item[
            "best_document_is_gold"
        ]

        for item
        in requirement_results

        if (
            item[
                "best_document_is_gold"
            ]
            is not None
        )
    ]


    context_document_ids = list(
        dict.fromkeys(
            item.document_id
            for item
            in context.items
        )
    )


    context_gold_recall = (
        gold_recall(
            predicted_ids=
                context_document_ids,

            gold_ids=
                gold_document_ids,
        )
    )


    # ========================================================
    # V2 decision for comparison
    # ========================================================

    v2_grade = (
        nodes.evidence_grader.grade(
            query=
                question,

            context=
                context,

            query_type=
                query_type,
        )
    )


    global_critical_terms = (
        nodes.evidence_grader
        ._critical_terms(
            question
        )
    )


    global_critical_coverage = (
        analyzer.critical_coverage(
            text=
                context.text,

            critical_terms=
                global_critical_terms,
        )
    )


    global_required_critical_coverage = (
        nodes.evidence_grader
        ._critical_coverage_requirement(
            query_type,
            global_critical_terms,
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
            context_gold_recall,

        "context_gold_complete": (
            (
                context_gold_recall
                ==
                1.0
            )
            if (
                context_gold_recall
                is not None
            )
            else None
        ),

        "v2_sufficient":
            v2_grade.sufficient,

        "v2_score":
            v2_grade.evidence_score,

        "v2_query_term_coverage":
            v2_grade.query_term_coverage,

        "requirement_count":
            len(
                requirement_results
            ),

        "requirement_mean_best_score":
            (
                round(
                    mean(
                        best_scores
                    ),
                    4,
                )
                if best_scores
                else None
            ),

        "requirement_min_best_score": (
            round(
                min(
                    best_scores
                ),
                4,
            )
            if best_scores
            else None
        ),

        "requirement_median_best_score": (
            round(
                statistics.median(
                    best_scores
                ),
                4,
            )
            if best_scores
            else None
        ),

        "requirement_max_best_score": (
            round(
                max(
                    best_scores
                ),
                4,
            )
            if best_scores
            else None
        ),

        "mean_top1_top2_margin": (
            round(
                mean(
                    margins
                ),
                4,
            )
            if margins
            else None
        ),

        "anchor_ok_fraction": (
            round(
                (
                    anchor_ok_count
                    /
                    len(
                        requirement_results
                    )
                ),
                4,
            )
            if requirement_results
            else 0.0
        ),

        "best_document_gold_fraction": (
            round(
                (
                    sum(
                        1
                        for value
                        in gold_best_flags
                        if value
                    )
                    /
                    len(
                        gold_best_flags
                    )
                ),
                4,
            )
            if gold_best_flags
            else None
        ),

        "global_critical_coverage":
            round(
                global_critical_coverage,
                4,
            ),

        "global_required_critical_coverage":
            round(
                global_required_critical_coverage,
                4,
            ),

        "global_anchor_ok": (
            global_critical_coverage
            >=
            global_required_critical_coverage
        ),

        "requirements":
            requirement_results,
    }


# ============================================================
# Aggregate semantic feature distributions
# ============================================================

def summarize_group(
    records,
):

    if not records:

        return {}


    return {
        "count":
            len(
                records
            ),

        "mean_requirement_mean_best_score":
            mean(
                [
                    record[
                        "requirement_mean_best_score"
                    ]
                    for record
                    in records
                ]
            ),

        "mean_requirement_min_best_score":
            mean(
                [
                    record[
                        "requirement_min_best_score"
                    ]
                    for record
                    in records
                ]
            ),

        "mean_requirement_median_best_score":
            mean(
                [
                    record[
                        "requirement_median_best_score"
                    ]
                    for record
                    in records
                ]
            ),

        "mean_anchor_ok_fraction":
            mean(
                [
                    record[
                        "anchor_ok_fraction"
                    ]
                    for record
                    in records
                ]
            ),

        "mean_best_document_gold_fraction":
            mean(
                [
                    record[
                        "best_document_gold_fraction"
                    ]
                    for record
                    in records
                ]
            ),

        "mean_top1_top2_margin":
            mean(
                [
                    record[
                        "mean_top1_top2_margin"
                    ]
                    for record
                    in records
                ]
            ),
    }


# ============================================================
# Exploratory threshold sweep
#
# IMPORTANT:
#
# This does NOT choose a production threshold.
#
# Smoke-20 is too small for calibration.
#
# Purpose:
#
# determine whether ANY semantic decision boundary appears
# promising enough to justify a larger frozen calibration.
# ============================================================

def semantic_threshold_sweep(
    records,
):

    requirement_scores = []


    for record in records:

        for requirement in (
            record[
                "requirements"
            ]
        ):

            score = (
                requirement[
                    "best_score"
                ]
            )


            if score is not None:

                requirement_scores.append(
                    float(
                        score
                    )
                )


    if not requirement_scores:

        return []


    unique_scores = sorted(
        set(
            requirement_scores
        )
    )


    # ========================================================
    # Use observed decision boundaries rather than inventing
    # arbitrary values such as 0, 2, 5, etc.
    # ========================================================

    thresholds = []


    if len(
        unique_scores
    ) == 1:

        thresholds = [
            unique_scores[
                0
            ]
        ]


    else:

        thresholds.append(
            unique_scores[
                0
            ]
            -
            0.001
        )


        for (
            left,
            right,
        ) in zip(
            unique_scores,
            unique_scores[
                1:
            ],
        ):

            thresholds.append(
                (
                    left
                    +
                    right
                )
                /
                2
            )


        thresholds.append(
            unique_scores[
                -1
            ]
            +
            0.001
        )


    required_fractions = [
        0.50,
        0.67,
        0.75,
        1.00,
    ]


    modes = [
        "semantic_only",
        "semantic_plus_local_anchor",
    ]


    results = []


    for threshold in thresholds:

        for required_fraction in (
            required_fractions
        ):

            for mode in modes:

                decisions = {}


                for record in records:

                    requirements = (
                        record[
                            "requirements"
                        ]
                    )


                    supported = 0


                    for requirement in requirements:

                        score_ok = (

                            requirement[
                                "best_score"
                            ]

                            is not None

                            and

                            requirement[
                                "best_score"
                            ]

                            >=

                            threshold
                        )


                        if (
                            mode
                            ==
                            "semantic_plus_local_anchor"
                        ):

                            requirement_ok = (

                                score_ok

                                and

                                requirement[
                                    "best_anchor_ok"
                                ]
                            )


                        else:

                            requirement_ok = (
                                score_ok
                            )


                        if requirement_ok:

                            supported += 1


                    required_count = max(
                        1,
                        math.ceil(
                            len(
                                requirements
                            )
                            *
                            required_fraction
                        ),
                    )


                    sufficient = (

                        supported

                        >=

                        required_count
                    )


                    decisions[
                        record[
                            "id"
                        ]
                    ] = (
                        sufficient
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


                answerable_accepts = sum(

                    1

                    for record
                    in answerable

                    if decisions[
                        record[
                            "id"
                        ]
                    ]
                )


                null_false_accepts = sum(

                    1

                    for record
                    in null_examples

                    if decisions[
                        record[
                            "id"
                        ]
                    ]
                )


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

                        not decisions[
                            record[
                                "id"
                            ]
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

                        not decisions[
                            record[
                                "id"
                            ]
                        ]
                    )
                )


                results.append(
                    {
                        "threshold":
                            round(
                                threshold,
                                4,
                            ),

                        "required_fraction":
                            required_fraction,

                        "mode":
                            mode,

                        "answerable_accepts":
                            answerable_accepts,

                        "answerable_accept_rate":
                            round(
                                (
                                    answerable_accepts
                                    /
                                    len(
                                        answerable
                                    )
                                ),
                                4,
                            ),

                        "null_false_accepts":
                            null_false_accepts,

                        "null_rejects":
                            (
                                len(
                                    null_examples
                                )
                                -
                                null_false_accepts
                            ),

                        "complete_gold_context_rejected":
                            complete_gold_rejected,

                        "high_gold_context_rejected":
                            high_gold_rejected,
                    }
                )


    # ========================================================
    # Rank diagnostically:
    #
    # 1. Prefer <= current V2 null false accepts.
    # 2. Maximize answerable acceptance.
    # 3. Minimize complete-gold rejection.
    # 4. Minimize high-gold rejection.
    #
    # Again: NOT production tuning.
    # ========================================================

    acceptable = [
        result
        for result
        in results
        if (
            result[
                "null_false_accepts"
            ]
            <=
            1
        )
    ]


    acceptable.sort(
        key=lambda result: (
            -result[
                "answerable_accepts"
            ],
            result[
                "complete_gold_context_rejected"
            ],
            result[
                "high_gold_context_rejected"
            ],
            result[
                "null_false_accepts"
            ],
        )
    )


    return (
        acceptable[
            :20
        ]
    )


# ============================================================
# Main
# ============================================================

def main():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        examples = json.load(
            file
        )


    nodes = (
        RAGNodes()
    )


    analyzer = (
        SemanticRequirementAnalyzer(
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
                    f"{index}/"
                    f"{len(examples)} | "
                    f"{example['question_type']}"
                )
            )


            print(
                example[
                    "question"
                ]
            )


            record = (
                run_case(
                    nodes=
                        nodes,

                    analyzer=
                        analyzer,

                    example=
                        example,
                )
            )


            records.append(
                record
            )


            print(
                "\nContext gold recall:",
                record[
                    "context_gold_recall"
                ],
            )


            print(
                (
                    "V2: "
                    f"sufficient="
                    f"{record['v2_sufficient']} "
                    f"score="
                    f"{record['v2_score']:.4f}"
                )
            )


            print(
                (
                    "Semantic example features: "
                    f"mean="
                    f"{record['requirement_mean_best_score']} "
                    f"min="
                    f"{record['requirement_min_best_score']} "
                    f"median="
                    f"{record['requirement_median_best_score']} "
                    f"anchor_fraction="
                    f"{record['anchor_ok_fraction']}"
                )
            )


            for (
                req_index,
                requirement,
            ) in enumerate(
                record[
                    "requirements"
                ],
                start=1,
            ):

                print(
                    (
                        f"\n  R{req_index}: "
                        f"{requirement['requirement']}"
                    )
                )


                print(
                    (
                        "     best_score="
                        f"{requirement['best_score']} "
                        "margin="
                        f"{requirement['score_margin']} "
                        "best_doc="
                        f"{requirement['best_document_id']} "
                        "gold="
                        f"{requirement['best_document_is_gold']} "
                        "anchor_ok="
                        f"{requirement['best_anchor_ok']}"
                    )
                )


    finally:

        nodes.close()


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


    complete_gold = [
        record
        for record
        in answerable
        if (
            record[
                "context_gold_complete"
            ]
            is True
        )
    ]


    v2_false_reject_complete = [
        record
        for record
        in complete_gold
        if not (
            record[
                "v2_sufficient"
            ]
        )
    ]


    summary = {
        "answerable":
            summarize_group(
                answerable
            ),

        "null":
            summarize_group(
                null_examples
            ),

        "complete_gold_answerable":
            summarize_group(
                complete_gold
            ),

        "v2_false_reject_complete_gold":
            summarize_group(
                v2_false_reject_complete
            ),

        "exploratory_best_semantic_policies":
            semantic_threshold_sweep(
                records
            ),
    }


    print(
        "\n\n"
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