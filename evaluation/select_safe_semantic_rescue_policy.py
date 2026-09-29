import json
import math

from collections import defaultdict
from pathlib import Path


INPUT_PATH = Path(
    "evaluation/results/"
    "semantic_evidence_gate_calibration.json"
)

OUTPUT_PATH = Path(
    "evaluation/results/"
    "strict_safe_semantic_rescue_policy.json"
)


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
# Semantic decision
# ============================================================

def semantic_decision(
    record,
    *,
    threshold,
    required_fraction,
    mode,
):

    requirements = record[
        "requirements"
    ]


    if not requirements:
        return False


    supported = 0


    for requirement in requirements:

        score = requirement[
            "best_score"
        ]


        score_ok = (
            score is not None
            and
            score >= threshold
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

            requirement_ok = score_ok


        if requirement_ok:
            supported += 1


    required_count = max(
        1,
        math.ceil(
            len(requirements)
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
# Hybrid decision
#
# V2 is primary.
#
# Semantic logic ONLY rescues V2 rejections.
# ============================================================

def hybrid_decision(
    record,
    *,
    threshold,
    required_fraction,
    mode,
):

    if record[
        "v2_sufficient"
    ]:

        return True


    return semantic_decision(
        record,
        threshold=threshold,
        required_fraction=(
            required_fraction
        ),
        mode=mode,
    )


# ============================================================
# Candidate thresholds
#
# Calibration only.
# ============================================================

def observed_thresholds(
    records,
):

    scores = []


    for record in records:

        if record[
            "v2_sufficient"
        ]:

            continue


        for requirement in record[
            "requirements"
        ]:

            score = requirement[
                "best_score"
            ]


            if score is not None:

                scores.append(
                    float(score)
                )


    values = sorted(
        set(scores)
    )


    if not values:
        return []


    thresholds = [
        values[0] - 0.001
    ]


    for left, right in zip(
        values,
        values[1:],
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
        values[-1] + 0.001
    )


    return thresholds


# ============================================================
# V2 baseline
# ============================================================

def baseline_metrics(
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


    null_records = [
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

        if record[
            "v2_sufficient"
        ]
    )


    null_false_ids = [
        record[
            "id"
        ]

        for record
        in null_records

        if record[
            "v2_sufficient"
        ]
    ]


    complete_rejected = sum(
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
                "v2_sufficient"
            ]
        )
    )


    high_rejected = sum(
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
                "v2_sufficient"
            ]
        )
    )


    return {
        "answerable_total":
            len(answerable),

        "answerable_accepts":
            answerable_accepts,

        "answerable_accept_rate": (
            answerable_accepts
            /
            len(answerable)
        ),

        "null_total":
            len(null_records),

        "null_false_accepts":
            len(null_false_ids),

        "null_false_accept_ids":
            null_false_ids,

        "null_reject_rate": (
            (
                len(null_records)
                -
                len(null_false_ids)
            )
            /
            len(null_records)
        ),

        "complete_gold_context_rejected":
            complete_rejected,

        "high_gold_context_rejected":
            high_rejected,
    }


# ============================================================
# Evaluate semantic rescue policy
# ============================================================

def evaluate_policy(
    records,
    *,
    threshold,
    required_fraction,
    mode,
):

    base = baseline_metrics(
        records
    )


    base_null_false_ids = set(
        base[
            "null_false_accept_ids"
        ]
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


    null_records = [
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


    rescued = []


    for record in answerable:

        if record[
            "v2_sufficient"
        ]:

            continue


        if semantic_decision(
            record,
            threshold=threshold,
            required_fraction=(
                required_fraction
            ),
            mode=mode,
        ):

            rescued.append(
                record
            )


    # ========================================================
    # Rescue quality by dataset evidence recall
    # ========================================================

    complete_rescues = [
        record

        for record
        in rescued

        if (
            record[
                "context_gold_complete"
            ]
            is True
        )
    ]


    high_rescues = [
        record

        for record
        in rescued

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


    low_rescues = [
        record

        for record
        in rescued

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
        in rescued

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
        in rescued

        if (
            record[
                "context_gold_recall"
            ]
            ==
            0.0
        )
    ]


    # ========================================================
    # Null safety
    # ========================================================

    final_null_false_ids = []


    for record in null_records:

        if hybrid_decision(
            record,
            threshold=threshold,
            required_fraction=(
                required_fraction
            ),
            mode=mode,
        ):

            final_null_false_ids.append(
                record[
                    "id"
                ]
            )


    final_null_false_set = set(
        final_null_false_ids
    )


    added_null_false_ids = sorted(
        final_null_false_set
        -
        base_null_false_ids
    )


    # ========================================================
    # Final system-level metrics
    # ========================================================

    answerable_accepts = (
        base[
            "answerable_accepts"
        ]
        +
        len(rescued)
    )


    complete_rejected = (
        base[
            "complete_gold_context_rejected"
        ]
        -
        len(
            complete_rescues
        )
    )


    high_rejected = (
        base[
            "high_gold_context_rejected"
        ]
        -
        len(
            high_rescues
        )
    )


    high_evidence_precision = None


    if rescued:

        high_evidence_precision = (
            len(
                high_rescues
            )
            /
            len(
                rescued
            )
        )


    return {
        "threshold":
            threshold,

        "required_fraction":
            required_fraction,

        "mode":
            mode,

        # ----------------------------------------------------
        # Overall rescue
        # ----------------------------------------------------

        "rescued_total":
            len(rescued),

        "rescued_ids": [
            record[
                "id"
            ]
            for record
            in rescued
        ],

        # ----------------------------------------------------
        # Evidence-backed rescue quality
        # ----------------------------------------------------

        "complete_gold_rescues":
            len(
                complete_rescues
            ),

        "high_gold_rescues":
            len(
                high_rescues
            ),

        "low_gold_rescues":
            len(
                low_rescues
            ),

        "severe_miss_rescues":
            len(
                severe_miss_rescues
            ),

        "zero_gold_rescues":
            len(
                zero_gold_rescues
            ),

        "rescue_high_evidence_precision":
            high_evidence_precision,

        # ----------------------------------------------------
        # Diagnostic IDs
        # ----------------------------------------------------

        "low_gold_rescue_ids": [
            record[
                "id"
            ]
            for record
            in low_rescues
        ],

        "severe_miss_rescue_ids": [
            record[
                "id"
            ]
            for record
            in severe_miss_rescues
        ],

        "zero_gold_rescue_ids": [
            record[
                "id"
            ]
            for record
            in zero_gold_rescues
        ],

        # ----------------------------------------------------
        # Null safety
        # ----------------------------------------------------

        "null_false_accepts":
            len(
                final_null_false_ids
            ),

        "added_null_false_accepts":
            len(
                added_null_false_ids
            ),

        "added_null_false_accept_ids":
            added_null_false_ids,

        # ----------------------------------------------------
        # Final acceptance
        # ----------------------------------------------------

        "answerable_accepts":
            answerable_accepts,

        "answerable_accept_rate": (
            answerable_accepts
            /
            base[
                "answerable_total"
            ]
        ),

        "complete_gold_context_rejected":
            complete_rejected,

        "high_gold_context_rejected":
            high_rejected,
    }


# ============================================================
# Strict calibration
#
# HARD safety constraints:
#
#   added null errors  == 0
#   zero-gold rescues  == 0
#   severe-miss rescue == 0
#
# Among safe candidates:
#
#   maximize complete evidence rescues
#   maximize high evidence rescues
#   minimize low evidence rescues
#   prefer stricter threshold/fraction
# ============================================================

def calibrate_strict_policy(
    records,
):

    base = baseline_metrics(
        records
    )


    candidates = []


    for threshold in observed_thresholds(
        records
    ):

        for required_fraction in (
            REQUIRED_FRACTIONS
        ):

            for mode in MODES:

                result = evaluate_policy(
                    records,
                    threshold=threshold,
                    required_fraction=(
                        required_fraction
                    ),
                    mode=mode,
                )


                if (
                    result[
                        "rescued_total"
                    ]
                    ==
                    0
                ):

                    continue


                # =================================================
                # HARD SAFETY CONSTRAINTS
                # =================================================

                if (
                    result[
                        "added_null_false_accepts"
                    ]
                    !=
                    0
                ):

                    continue


                if (
                    result[
                        "zero_gold_rescues"
                    ]
                    !=
                    0
                ):

                    continue


                if (
                    result[
                        "severe_miss_rescues"
                    ]
                    !=
                    0
                ):

                    continue


                candidates.append(
                    result
                )


    if not candidates:

        raise RuntimeError(
            (
                "No semantic rescue policy satisfied "
                "strict calibration safety."
            )
        )


    def mode_preference(
        mode,
    ):

        if (
            mode
            ==
            "semantic_plus_local_anchor"
        ):

            return 0

        return 1


    candidates.sort(
        key=lambda item: (

            # -----------------------------------------------
            # First save the strongest known evidence cases.
            # -----------------------------------------------

            -item[
                "complete_gold_rescues"
            ],

            -item[
                "high_gold_rescues"
            ],

            # -----------------------------------------------
            # Penalize evidence-weak rescues.
            # -----------------------------------------------

            item[
                "low_gold_rescues"
            ],

            # -----------------------------------------------
            # Prefer conservative boundaries.
            # -----------------------------------------------

            -item[
                "threshold"
            ],

            -item[
                "required_fraction"
            ],

            mode_preference(
                item[
                    "mode"
                ]
            ),
        )
    )


    return {
        "baseline":
            base,

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
# Query-type breakdown
# ============================================================

def breakdown_by_type(
    records,
    policy,
):

    output = defaultdict(
        lambda: {
            "rescued_total": 0,
            "complete_gold_rescues": 0,
            "high_gold_rescues": 0,
            "low_gold_rescues": 0,
            "severe_miss_rescues": 0,
            "zero_gold_rescues": 0,
        }
    )


    for record in records:

        if (
            record[
                "question_type"
            ]
            ==
            "null_query"
        ):

            continue


        if record[
            "v2_sufficient"
        ]:

            continue


        rescued = semantic_decision(
            record,
            threshold=(
                policy[
                    "threshold"
                ]
            ),
            required_fraction=(
                policy[
                    "required_fraction"
                ]
            ),
            mode=(
                policy[
                    "mode"
                ]
            ),
        )


        if not rescued:
            continue


        question_type = record[
            "question_type"
        ]


        data = output[
            question_type
        ]


        data[
            "rescued_total"
        ] += 1


        recall = record[
            "context_gold_recall"
        ]


        if (
            record[
                "context_gold_complete"
            ]
            is True
        ):

            data[
                "complete_gold_rescues"
            ] += 1


        if (
            recall is not None
            and
            recall >= 0.75
        ):

            data[
                "high_gold_rescues"
            ] += 1

        else:

            data[
                "low_gold_rescues"
            ] += 1


        if (
            recall is not None
            and
            recall <= 0.25
        ):

            data[
                "severe_miss_rescues"
            ] += 1


        if recall == 0.0:

            data[
                "zero_gold_rescues"
            ] += 1


    return dict(
        output
    )


# ============================================================
# Pretty summary
# ============================================================

def print_summary(
    title,
    base,
    candidate,
):

    print(
        "\n"
        +
        "=" * 100
    )

    print(title)

    print(
        "=" * 100
    )


    print(
        "\nV2:"
    )

    print(
        (
            "  answerable: "
            f"{base['answerable_accepts']}/"
            f"{base['answerable_total']} "
            f"({base['answerable_accept_rate']:.4f})"
        )
    )

    print(
        (
            "  null false accepts: "
            f"{base['null_false_accepts']}"
        )
    )

    print(
        (
            "  complete-gold rejected: "
            f"{base['complete_gold_context_rejected']}"
        )
    )

    print(
        (
            "  high-gold rejected: "
            f"{base['high_gold_context_rejected']}"
        )
    )


    print(
        "\nV2 + STRICT SEMANTIC RESCUE:"
    )

    print(
        (
            "  answerable: "
            f"{candidate['answerable_accepts']}/"
            f"{base['answerable_total']} "
            f"({candidate['answerable_accept_rate']:.4f})"
        )
    )

    print(
        (
            "  rescued total: "
            f"{candidate['rescued_total']}"
        )
    )

    print(
        (
            "  complete-gold rescues: "
            f"{candidate['complete_gold_rescues']}"
        )
    )

    print(
        (
            "  high-gold rescues: "
            f"{candidate['high_gold_rescues']}"
        )
    )

    print(
        (
            "  low-gold rescues: "
            f"{candidate['low_gold_rescues']}"
        )
    )

    print(
        (
            "  severe-miss rescues: "
            f"{candidate['severe_miss_rescues']}"
        )
    )

    print(
        (
            "  zero-gold rescues: "
            f"{candidate['zero_gold_rescues']}"
        )
    )

    print(
        (
            "  rescue high-evidence precision: "
            f"{candidate['rescue_high_evidence_precision']}"
        )
    )

    print(
        (
            "  added null false accepts: "
            f"{candidate['added_null_false_accepts']}"
        )
    )

    print(
        (
            "  complete-gold rejected: "
            f"{candidate['complete_gold_context_rejected']}"
        )
    )

    print(
        (
            "  high-gold rejected: "
            f"{candidate['high_gold_context_rejected']}"
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(
            file
        )


    calibration_records = payload[
        "calibration_records"
    ]


    validation_records = payload[
        "validation_records"
    ]


    # ========================================================
    # Select using CALIBRATION ONLY
    # ========================================================

    calibration = (
        calibrate_strict_policy(
            calibration_records
        )
    )


    selected = calibration[
        "selected"
    ]


    policy = {
        "threshold":
            selected[
                "threshold"
            ],

        "required_fraction":
            selected[
                "required_fraction"
            ],

        "mode":
            selected[
                "mode"
            ],
    }


    # ========================================================
    # Validation:
    #
    # policy is now frozen.
    # ========================================================

    validation_base = (
        baseline_metrics(
            validation_records
        )
    )


    validation_candidate = (
        evaluate_policy(
            validation_records,
            threshold=(
                policy[
                    "threshold"
                ]
            ),
            required_fraction=(
                policy[
                    "required_fraction"
                ]
            ),
            mode=(
                policy[
                    "mode"
                ]
            ),
        )
    )


    output = {
        "selected_policy":
            policy,

        "calibration": {
            "v2":
                calibration[
                    "baseline"
                ],

            "strict_semantic_rescue":
                selected,

            "top_candidates":
                calibration[
                    "top_candidates"
                ],

            "by_question_type":
                breakdown_by_type(
                    calibration_records,
                    policy,
                ),
        },

        "validation": {
            "v2":
                validation_base,

            "strict_semantic_rescue":
                validation_candidate,

            "by_question_type":
                breakdown_by_type(
                    validation_records,
                    policy,
                ),
        },
    }


    print(
        "\nSELECTED STRICT SAFE POLICY:"
    )

    print(
        json.dumps(
            policy,
            indent=2,
        )
    )


    print_summary(
        title=
            "CALIBRATION",

        base=
            calibration[
                "baseline"
            ],

        candidate=
            selected,
    )


    print_summary(
        title=
            "HELD-OUT VALIDATION",

        base=
            validation_base,

        candidate=
            validation_candidate,
    )


    print(
        "\nVALIDATION BY QUESTION TYPE:"
    )

    print(
        json.dumps(
            output[
                "validation"
            ][
                "by_question_type"
            ],
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