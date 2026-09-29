import json
import math

from pathlib import Path


INPUT_PATH = Path(
    "evaluation/results/"
    "semantic_evidence_gate_calibration.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "semantic_rescue_gate_analysis.json"
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
# Hybrid:
#
# V2 accept
# OR
# semantic rescue
# ============================================================

def hybrid_decision(
    record,
    *,
    threshold,
    required_fraction,
    mode,
):

    if (
        record[
            "v2_sufficient"
        ]
    ):

        return True


    return semantic_decision(
        record,
        threshold=
            threshold,
        required_fraction=
            required_fraction,
        mode=
            mode,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
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


    accepted_answerable = [
        record
        for record
        in answerable
        if decision_function(
            record
        )
    ]


    null_false_accepts = [
        record
        for record
        in null_examples
        if decision_function(
            record
        )
    ]


    complete_gold_rejected = [
        record
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
    ]


    high_gold_rejected = [
        record
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
    ]


    return {
        "answerable_total":
            len(
                answerable
            ),

        "answerable_accepts":
            len(
                accepted_answerable
            ),

        "answerable_accept_rate": (
            len(
                accepted_answerable
            )
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
            len(
                null_false_accepts
            ),

        "null_rejects": (
            len(
                null_examples
            )
            -
            len(
                null_false_accepts
            )
        ),

        "null_reject_rate": (
            (
                len(
                    null_examples
                )
                -
                len(
                    null_false_accepts
                )
            )
            /
            len(
                null_examples
            )
        ),

        "complete_gold_context_rejected":
            len(
                complete_gold_rejected
            ),

        "high_gold_context_rejected":
            len(
                high_gold_rejected
            ),

        "accepted_answerable_ids": [
            record[
                "id"
            ]
            for record
            in accepted_answerable
        ],

        "null_false_accept_ids": [
            record[
                "id"
            ]
            for record
            in null_false_accepts
        ],
    }


# ============================================================
# Observed threshold boundaries
#
# Calibration only.
# ============================================================

def observed_thresholds(
    records,
):

    scores = []


    # --------------------------------------------------------
    # Only V2-rejected records matter for rescue threshold.
    # --------------------------------------------------------

    for record in records:

        if (
            record[
                "v2_sufficient"
            ]
        ):

            continue


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


# ============================================================
# Calibrate rescue gate
# ============================================================

def calibrate_rescue(
    records,
):

    v2_metrics = evaluate(
        records,
        decision_function=lambda record:
            record[
                "v2_sufficient"
            ],
    )


    baseline_null_false_accepts = (
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

        for required_fraction in (
            REQUIRED_FRACTIONS
        ):

            for mode in MODES:

                metrics = evaluate(
                    records,
                    decision_function=(
                        lambda record,
                        t=threshold,
                        f=required_fraction,
                        m=mode:

                        hybrid_decision(
                            record,
                            threshold=t,
                            required_fraction=f,
                            mode=m,
                        )
                    ),
                )


                # =================================================
                # HARD calibration safety condition:
                #
                # Semantic rescue may NOT introduce even one
                # additional null false accept.
                # =================================================

                if (
                    metrics[
                        "null_false_accepts"
                    ]
                    >
                    baseline_null_false_accepts
                ):

                    continue


                rescued_answerable = (

                    metrics[
                        "answerable_accepts"
                    ]

                    -

                    v2_metrics[
                        "answerable_accepts"
                    ]
                )


                candidates.append(
                    {
                        "threshold":
                            threshold,

                        "required_fraction":
                            required_fraction,

                        "mode":
                            mode,

                        "rescued_answerable":
                            rescued_answerable,

                        "metrics":
                            metrics,
                    }
                )


    if not candidates:

        raise RuntimeError(
            "No safe semantic rescue policy found."
        )


    # ========================================================
    # Selection:
    #
    # 1. maximize rescued answerable
    # 2. minimize complete-gold rejection
    # 3. minimize high-gold rejection
    # 4. prefer zero additional null errors
    # 5. prefer higher semantic threshold
    # ========================================================

    candidates.sort(
        key=lambda candidate: (
            -candidate[
                "rescued_answerable"
            ],

            candidate[
                "metrics"
            ][
                "complete_gold_context_rejected"
            ],

            candidate[
                "metrics"
            ][
                "high_gold_context_rejected"
            ],

            candidate[
                "metrics"
            ][
                "null_false_accepts"
            ],

            -candidate[
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
# Difference analysis
# ============================================================

def difference_report(
    records,
    *,
    threshold,
    required_fraction,
    mode,
):

    rescued_answerable = []

    added_null_false_accepts = []


    for record in records:

        v2 = (
            record[
                "v2_sufficient"
            ]
        )


        hybrid = (
            hybrid_decision(
                record,
                threshold=
                    threshold,
                required_fraction=
                    required_fraction,
                mode=
                    mode,
            )
        )


        if (
            not v2
            and
            hybrid
        ):

            if (
                record[
                    "question_type"
                ]
                ==
                "null_query"
            ):

                added_null_false_accepts.append(
                    record[
                        "id"
                    ]
                )

            else:

                rescued_answerable.append(
                    {
                        "id":
                            record[
                                "id"
                            ],

                        "question_type":
                            record[
                                "question_type"
                            ],

                        "context_gold_recall":
                            record[
                                "context_gold_recall"
                            ],

                        "context_gold_complete":
                            record[
                                "context_gold_complete"
                            ],
                    }
                )


    return {
        "rescued_answerable_count":
            len(
                rescued_answerable
            ),

        "rescued_answerable":
            rescued_answerable,

        "added_null_false_accept_count":
            len(
                added_null_false_accepts
            ),

        "added_null_false_accept_ids":
            added_null_false_accepts,
    }


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


    calibration_records = (
        payload[
            "calibration_records"
        ]
    )


    validation_records = (
        payload[
            "validation_records"
        ]
    )


    # ========================================================
    # Select policy using CALIBRATION ONLY.
    # ========================================================

    calibration = (
        calibrate_rescue(
            calibration_records
        )
    )


    selected = (
        calibration[
            "selected"
        ]
    )


    threshold = (
        selected[
            "threshold"
        ]
    )


    required_fraction = (
        selected[
            "required_fraction"
        ]
    )


    mode = (
        selected[
            "mode"
        ]
    )


    # ========================================================
    # Validation:
    #
    # no threshold modification after this point.
    # ========================================================

    validation_v2 = (
        evaluate(
            validation_records,
            decision_function=lambda record:
                record[
                    "v2_sufficient"
                ],
        )
    )


    validation_hybrid = (
        evaluate(
            validation_records,
            decision_function=lambda record:
                hybrid_decision(
                    record,
                    threshold=
                        threshold,
                    required_fraction=
                        required_fraction,
                    mode=
                        mode,
                ),
        )
    )


    calibration_difference = (
        difference_report(
            calibration_records,
            threshold=
                threshold,
            required_fraction=
                required_fraction,
            mode=
                mode,
        )
    )


    validation_difference = (
        difference_report(
            validation_records,
            threshold=
                threshold,
            required_fraction=
                required_fraction,
            mode=
                mode,
        )
    )


    output = {
        "selected_policy": {
            "threshold":
                threshold,

            "required_fraction":
                required_fraction,

            "mode":
                mode,
        },

        "calibration": {
            "v2":
                calibration[
                    "v2"
                ],

            "hybrid":
                selected[
                    "metrics"
                ],

            "difference":
                calibration_difference,

            "top_candidates":
                calibration[
                    "top_candidates"
                ],
        },

        "validation": {
            "v2":
                validation_v2,

            "hybrid":
                validation_hybrid,

            "difference":
                validation_difference,
        },
    }


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "SELECTED SEMANTIC RESCUE POLICY"
    )


    print(
        "=" * 100
    )


    print(
        json.dumps(
            output[
                "selected_policy"
            ],
            indent=2,
        )
    )


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "CALIBRATION"
    )


    print(
        "=" * 100
    )


    print(
        "\nV2:"
    )

    print(
        json.dumps(
            output[
                "calibration"
            ][
                "v2"
            ],
            indent=2,
        )
    )


    print(
        "\nHybrid:"
    )

    print(
        json.dumps(
            output[
                "calibration"
            ][
                "hybrid"
            ],
            indent=2,
        )
    )


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "HELD-OUT VALIDATION"
    )


    print(
        "=" * 100
    )


    print(
        "\nV2:"
    )

    print(
        json.dumps(
            validation_v2,
            indent=2,
        )
    )


    print(
        "\nV2 + Semantic Rescue:"
    )

    print(
        json.dumps(
            validation_hybrid,
            indent=2,
        )
    )


    print(
        "\nDifference:"
    )

    print(
        json.dumps(
            validation_difference,
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