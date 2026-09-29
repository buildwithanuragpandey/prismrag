import json
import math
import random

from collections import defaultdict
from pathlib import Path

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)

from adaptive_agentic_rag.retrieval.query_decomposer import (
    QueryDecomposer,
)


SOURCE_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)

SMOKE_MANIFEST_PATH = Path(
    "evaluation/datasets/"
    "frozen_e2e_smoke_20_manifest.json"
)

OUTPUT_PATH = Path(
    "evaluation/results/"
    "semantic_evidence_gate_calibration.json"
)

SPLIT_MANIFEST_PATH = Path(
    "evaluation/datasets/"
    "semantic_gate_calibration_split.json"
)


SEED = 2601


REQUIRED_FRACTIONS = [
    0.50,
    0.67,
    0.75,
    1.00,
]


MODES = [
    "semantic_only",
    "semantic_plus_local_anchor",
]


# ============================================================
# Dataset
# ============================================================

def load_examples():

    with open(
        SOURCE_PATH,
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
        "Could not find examples in frozen_eval_500.json"
    )


def load_smoke_ids():

    with open(
        SMOKE_MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        manifest = json.load(
            file
        )


    return set(
        manifest[
            "ids"
        ]
    )


# ============================================================
# Deterministic stratified split
# ============================================================

def stratified_split(
    examples,
):

    rng = random.Random(
        SEED
    )


    grouped = defaultdict(
        list
    )


    for example in examples:

        grouped[
            example[
                "question_type"
            ]
        ].append(
            example
        )


    calibration = []

    validation = []

    distribution = {}


    for question_type in sorted(
        grouped
    ):

        items = sorted(
            grouped[
                question_type
            ],
            key=lambda item:
                item[
                    "id"
                ],
        )


        rng.shuffle(
            items
        )


        midpoint = (
            len(items)
            //
            2
        )


        calibration_items = (
            items[
                :midpoint
            ]
        )


        validation_items = (
            items[
                midpoint:
            ]
        )


        calibration.extend(
            calibration_items
        )


        validation.extend(
            validation_items
        )


        distribution[
            question_type
        ] = {
            "total":
                len(
                    items
                ),

            "calibration":
                len(
                    calibration_items
                ),

            "validation":
                len(
                    validation_items
                ),
        }


    calibration.sort(
        key=lambda item:
            item[
                "id"
            ]
    )


    validation.sort(
        key=lambda item:
            item[
                "id"
            ]
    )


    return (
        calibration,
        validation,
        distribution,
    )


# ============================================================
# Misc helpers
# ============================================================

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
# Semantic requirement feature extractor
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


    def extract_requirements(
        self,
        query,
    ):

        output = (
            self.decomposer.decompose(
                query
            )
        )


        if not output:

            return [
                query
            ]


        if len(
            output
        ) > 1:

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


        cleaned = []

        seen = set()


        for requirement in requirements:

            normalized = (
                " ".join(
                    requirement.split()
                )
            )


            key = (
                normalized.lower()
            )


            if (
                not normalized
                or
                key in seen
            ):

                continue


            seen.add(
                key
            )


            cleaned.append(
                normalized
            )


        return (
            cleaned
            or
            [
                query
            ]
        )


    @staticmethod
    def build_candidates(
        context,
    ):

        output = []


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


            output.append(
                {
                    "id":
                        f"context_{index}",

                    "text":
                        "\n".join(
                            parts
                        ),

                    "document_id":
                        item.document_id,

                    "chunk_id":
                        item.chunk_id,

                    "citation_id":
                        item.citation_id,
                }
            )


        return output


    def critical_coverage(
        self,
        text,
        critical_terms,
    ):

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
            len(
                matched
            )
            /
            len(
                critical_terms
            )
        )


    def analyze_requirement(
        self,
        requirement,
        candidates,
        query_type,
    ):

        critical_terms = (
            self.evidence_grader
            ._critical_terms(
                requirement
            )
        )


        required_critical = (
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
                "best_score":
                    None,

                "best_anchor_ok":
                    False,

                "best_document_id":
                    None,
            }


        best = (
            ranked[
                0
            ]
        )


        coverage = (
            self.critical_coverage(
                best[
                    "text"
                ],
                critical_terms,
            )
        )


        return {
            "best_score":
                float(
                    best[
                        "rerank_score"
                    ]
                ),

            "best_anchor_ok": (
                coverage
                >=
                required_critical
            ),

            "best_document_id":
                best[
                    "document_id"
                ],
        }


# ============================================================
# Extract feature record from production retrieval/context
# ============================================================

def extract_record(
    nodes,
    analyzer,
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


    query_type = (
        state[
            "query_type"
        ]
    )


    context = (
        state[
            "context"
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


    semantic_requirements = []


    for requirement in requirements:

        result = (
            analyzer.analyze_requirement(
                requirement=
                    requirement,

                candidates=
                    candidates,

                query_type=
                    query_type,
            )
        )


        semantic_requirements.append(
            {
                "text":
                    requirement,

                **result,
            }
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
            context_document_ids,
            gold_document_ids,
        )
    )


    return {
        "id":
            example[
                "id"
            ],

        "question_type":
            example[
                "question_type"
            ],

        "router_query_type":
            query_type,

        "gold_document_ids":
            gold_document_ids,

        "context_document_ids":
            context_document_ids,

        "context_gold_recall":
            context_recall,

        "context_gold_complete": (
            (
                context_recall
                ==
                1.0
            )
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

        "requirements":
            semantic_requirements,
    }


# ============================================================
# Semantic policy
# ============================================================

def semantic_decision(
    record,
    *,
    threshold,
    required_fraction,
    mode,
):

    requirements = (
        record[
            "requirements"
        ]
    )


    if not requirements:

        return False


    supported = 0


    for requirement in requirements:

        score = (
            requirement[
                "best_score"
            ]
        )


        score_ok = (

            score
            is not None

            and

            score
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


    return (
        supported
        >=
        required_count
    )


# ============================================================
# Evaluate decisions
# ============================================================

def evaluate_policy(
    records,
    decision_function,
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


    answerable_accepts = sum(

        1

        for record
        in answerable

        if decision_function(
            record
        )
    )


    null_false_accepts = sum(

        1

        for record
        in null_examples

        if decision_function(
            record
        )
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

            not decision_function(
                record
            )
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

            not decision_function(
                record
            )
        )
    )


    return {
        "answerable_total":
            len(
                answerable
            ),

        "answerable_accepts":
            answerable_accepts,

        "answerable_accept_rate": (
            answerable_accepts
            /
            len(
                answerable
            )
        ),

        "null_total":
            len(
                null_examples
            ),

        "null_false_accepts":
            null_false_accepts,

        "null_rejects": (
            len(
                null_examples
            )
            -
            null_false_accepts
        ),

        "null_reject_rate": (
            (
                len(
                    null_examples
                )
                -
                null_false_accepts
            )
            /
            len(
                null_examples
            )
        ),

        "complete_gold_context_rejected":
            complete_gold_rejected,

        "high_gold_context_rejected":
            high_gold_rejected,
    }


# ============================================================
# Calibration search
# ============================================================

def observed_thresholds(
    records,
):

    scores = []


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

                scores.append(
                    float(
                        score
                    )
                )


    values = sorted(
        set(
            scores
        )
    )


    if not values:

        return []


    thresholds = [
        values[
            0
        ]
        -
        0.001
    ]


    for (
        left,
        right,
    ) in zip(
        values,
        values[
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
        values[
            -1
        ]
        +
        0.001
    )


    return thresholds


def calibrate(
    records,
):

    v2_metrics = (
        evaluate_policy(
            records,
            decision_function=lambda record:
                record[
                    "v2_sufficient"
                ],
        )
    )


    # --------------------------------------------------------
    # Candidate may not exceed V2's null false-accept count
    # on calibration.
    # --------------------------------------------------------

    max_null_false_accepts = (
        v2_metrics[
            "null_false_accepts"
        ]
    )


    candidates = []


    for threshold in (
        observed_thresholds(
            records
        )
    ):

        for fraction in (
            REQUIRED_FRACTIONS
        ):

            for mode in MODES:

                metrics = (
                    evaluate_policy(
                        records,
                        decision_function=lambda record,
                        t=threshold,
                        f=fraction,
                        m=mode:
                            semantic_decision(
                                record,
                                threshold=t,
                                required_fraction=f,
                                mode=m,
                            ),
                    )
                )


                if (
                    metrics[
                        "null_false_accepts"
                    ]
                    >
                    max_null_false_accepts
                ):

                    continue


                candidates.append(
                    {
                        "threshold":
                            threshold,

                        "required_fraction":
                            fraction,

                        "mode":
                            mode,

                        "metrics":
                            metrics,
                    }
                )


    if not candidates:

        raise RuntimeError(
            "No semantic candidate satisfied calibration safety."
        )


    # ========================================================
    # Selection:
    #
    # 1. maximize answerable acceptance
    # 2. minimize complete-gold false rejection
    # 3. minimize high-gold rejection
    # 4. minimize null false accepts
    # 5. prefer higher threshold when otherwise identical
    # ========================================================

    candidates.sort(
        key=lambda item: (
            -item[
                "metrics"
            ][
                "answerable_accepts"
            ],

            item[
                "metrics"
            ][
                "complete_gold_context_rejected"
            ],

            item[
                "metrics"
            ][
                "high_gold_context_rejected"
            ],

            item[
                "metrics"
            ][
                "null_false_accepts"
            ],

            -item[
                "threshold"
            ],
        )
    )


    return {
        "v2":
            v2_metrics,

        "selected":
            candidates[
                0
            ],

        "top_candidates":
            candidates[
                :20
            ],
    }


# ============================================================
# Split execution
# ============================================================

def extract_split_features(
    nodes,
    analyzer,
    examples,
    label,
):

    records = []


    total = len(
        examples
    )


    for (
        index,
        example,
    ) in enumerate(
        examples,
        start=1,
    ):

        print(
            (
                f"[{label}] "
                f"{index}/{total} "
                f"{example['question_type']} "
                f"{example['id']}"
            )
        )


        records.append(
            extract_record(
                nodes=
                    nodes,

                analyzer=
                    analyzer,

                example=
                    example,
            )
        )


    return records


# ============================================================
# Main
# ============================================================

def main():

    examples = (
        load_examples()
    )


    smoke_ids = (
        load_smoke_ids()
    )


    untouched = [
        example

        for example
        in examples

        if (
            example[
                "id"
            ]
            not in
            smoke_ids
        )
    ]


    if len(
        untouched
    ) != 480:

        raise ValueError(
            (
                "Expected 480 untouched examples, "
                f"found {len(untouched)}."
            )
        )


    (
        calibration_examples,
        validation_examples,
        distribution,
    ) = stratified_split(
        untouched
    )


    split_manifest = {
        "source":
            str(
                SOURCE_PATH
            ),

        "excluded_smoke_ids":
            sorted(
                smoke_ids
            ),

        "seed":
            SEED,

        "distribution":
            distribution,

        "calibration_ids": [
            example[
                "id"
            ]
            for example
            in calibration_examples
        ],

        "validation_ids": [
            example[
                "id"
            ]
            for example
            in validation_examples
        ],
    }


    SPLIT_MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        SPLIT_MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            split_manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


    print(
        "\nCalibration:",
        len(
            calibration_examples
        )
    )


    print(
        "Validation:",
        len(
            validation_examples
        )
    )


    print(
        "\nDistribution:"
    )


    print(
        json.dumps(
            distribution,
            indent=2,
        )
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


    try:

        calibration_records = (
            extract_split_features(
                nodes=
                    nodes,

                analyzer=
                    analyzer,

                examples=
                    calibration_examples,

                label=
                    "CAL",
            )
        )


        calibration_result = (
            calibrate(
                calibration_records
            )
        )


        selected = (
            calibration_result[
                "selected"
            ]
        )


        print(
            "\n"
            +
            "=" * 100
        )


        print(
            "CALIBRATION SELECTED POLICY"
        )


        print(
            "=" * 100
        )


        print(
            json.dumps(
                selected,
                indent=2,
            )
        )


        # ====================================================
        # IMPORTANT:
        #
        # Validation is processed only AFTER policy selection.
        # ====================================================

        validation_records = (
            extract_split_features(
                nodes=
                    nodes,

                analyzer=
                    analyzer,

                examples=
                    validation_examples,

                label=
                    "VAL",
            )
        )


    finally:

        nodes.close()


    v2_validation = (
        evaluate_policy(
            validation_records,
            decision_function=lambda record:
                record[
                    "v2_sufficient"
                ],
        )
    )


    semantic_validation = (
        evaluate_policy(
            validation_records,
            decision_function=lambda record:
                semantic_decision(
                    record,
                    threshold=(
                        selected[
                            "threshold"
                        ]
                    ),
                    required_fraction=(
                        selected[
                            "required_fraction"
                        ]
                    ),
                    mode=(
                        selected[
                            "mode"
                        ]
                    ),
                ),
        )
    )


    output = {
        "split_manifest":
            str(
                SPLIT_MANIFEST_PATH
            ),

        "calibration": {
            "v2":
                calibration_result[
                    "v2"
                ],

            "selected":
                selected,

            "top_candidates":
                calibration_result[
                    "top_candidates"
                ],
        },

        "validation": {
            "v2":
                v2_validation,

            "semantic":
                semantic_validation,
        },

        "calibration_records":
            calibration_records,

        "validation_records":
            validation_records,
    }


    print(
        "\n\n"
        +
        "=" * 100
    )


    print(
        "FINAL HELD-OUT VALIDATION"
    )


    print(
        "=" * 100
    )


    print(
        "\nV2:"
    )


    print(
        json.dumps(
            v2_validation,
            indent=2,
        )
    )


    print(
        "\nSemantic candidate:"
    )


    print(
        json.dumps(
            semantic_validation,
            indent=2,
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