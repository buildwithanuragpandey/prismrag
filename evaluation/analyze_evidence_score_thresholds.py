import json
from pathlib import Path


INPUT_PATH = Path(
    "evaluation/results/"
    "evidence_gate_v2_500.json"
)


THRESHOLDS = [
    round(
        value / 100,
        2,
    )
    for value
    in range(
        75,
        91,
    )
]


# ============================================================
# Hard blockers
# ============================================================

def has_hard_blocker(
    attempt: dict,
) -> bool:

    reasons = (
        attempt.get(
            "evidence_reasons",
            [],
        )
        or []
    )


    hard_prefixes = (
        "No evidence was retrieved.",
        "Insufficient document diversity:",
        "Too few evidence chunks:",
        "Critical query anchors are missing",
    )


    return any(

        reason.startswith(
            hard_prefixes
        )

        for reason
        in reasons
    )


# ============================================================
# Policy
# ============================================================

def passes(
    attempt: dict,
    threshold: float,
) -> bool:

    if has_hard_blocker(
        attempt
    ):
        return False


    return (
        float(
            attempt.get(
                "evidence_score",
                0.0,
            )
        )
        >=
        threshold
    )


# ============================================================
# Safe ratio
# ============================================================

def ratio(
    numerator: int,
    denominator: int,
):

    if denominator == 0:
        return None

    return (
        numerator
        /
        denominator
    )


# ============================================================
# Evaluate one threshold
# ============================================================

def evaluate_threshold(
    records: list[dict],
    threshold: float,
):

    rows = []


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Use ATTEMPT 0 only.
    #
    # This isolates EvidenceGrader from QueryRewriter.
    # Otherwise changing the threshold also changes which
    # examples would receive a second retrieval attempt.
    # --------------------------------------------------------

    for record in records:

        attempt = (
            record[
                "attempts"
            ][0]
        )


        rows.append(
            {
                "record":
                    record,

                "attempt":
                    attempt,

                "passed":
                    passes(
                        attempt,
                        threshold,
                    ),
            }
        )


    answerable = [
        row
        for row in rows
        if row[
            "record"
        ][
            "is_answerable"
        ]
    ]


    null_examples = [
        row
        for row in rows
        if not row[
            "record"
        ][
            "is_answerable"
        ]
    ]


    # ========================================================
    # Complete dataset-evidence context
    # ========================================================

    complete = [
        row
        for row in answerable
        if (
            row[
                "attempt"
            ][
                "context_gold_recall"
            ]
            ==
            1.0
        )
    ]


    complete_rejected = [
        row
        for row in complete
        if not row[
            "passed"
        ]
    ]


    # ========================================================
    # High context recall
    # ========================================================

    high_context = [
        row
        for row in answerable
        if (
            row[
                "attempt"
            ][
                "context_gold_recall"
            ]
            is not None
            and
            row[
                "attempt"
            ][
                "context_gold_recall"
            ]
            >=
            0.75
        )
    ]


    high_rejected = [
        row
        for row in high_context
        if not row[
            "passed"
        ]
    ]


    # ========================================================
    # Low context recall
    # ========================================================

    low_context = [
        row
        for row in answerable
        if (
            row[
                "attempt"
            ][
                "context_gold_recall"
            ]
            is not None
            and
            row[
                "attempt"
            ][
                "context_gold_recall"
            ]
            <=
            0.25
        )
    ]


    low_accepted = [
        row
        for row in low_context
        if row[
            "passed"
        ]
    ]


    answerable_passed = sum(

        1
        for row
        in answerable
        if row[
            "passed"
        ]
    )


    null_passed = sum(

        1
        for row
        in null_examples
        if row[
            "passed"
        ]
    )


    return {
        "threshold":
            threshold,

        "answerable_pass_count":
            answerable_passed,

        "answerable_pass_rate":
            ratio(
                answerable_passed,
                len(
                    answerable
                ),
            ),

        "null_pass_count":
            null_passed,

        "null_pass_rate":
            ratio(
                null_passed,
                len(
                    null_examples
                ),
            ),

        "complete_context_count":
            len(
                complete
            ),

        "complete_context_rejection_rate":
            ratio(
                len(
                    complete_rejected
                ),
                len(
                    complete
                ),
            ),

        "high_context_rejection_rate":
            ratio(
                len(
                    high_rejected
                ),
                len(
                    high_context
                ),
            ),

        "low_context_acceptance_rate":
            ratio(
                len(
                    low_accepted
                ),
                len(
                    low_context
                ),
            ),
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


    records = payload[
        "records"
    ]


    results = [

        evaluate_threshold(
            records,
            threshold,
        )

        for threshold
        in THRESHOLDS
    ]


    print(
        "=" * 118
    )

    print(
        "EVIDENCE SCORE THRESHOLD SWEEP — ATTEMPT 0"
    )

    print(
        "=" * 118
    )


    header = (
        f"{'THR':>6s}"
        f"{'ANS PASS':>11s}"
        f"{'ANS N':>8s}"
        f"{'NULL PASS':>12s}"
        f"{'NULL N':>8s}"
        f"{'COMP REJ':>11s}"
        f"{'HIGH REJ':>11s}"
        f"{'LOW ACC':>10s}"
    )


    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )


    for result in results:

        print(
            f"{result['threshold']:6.2f}"

            f"{result['answerable_pass_rate']:11.4f}"

            f"{result['answerable_pass_count']:8d}"

            f"{result['null_pass_rate']:12.4f}"

            f"{result['null_pass_count']:8d}"

            f"{result['complete_context_rejection_rate']:11.4f}"

            f"{result['high_context_rejection_rate']:11.4f}"

            f"{result['low_context_acceptance_rate']:10.4f}"
        )


    # ========================================================
    # Constraint-based recommendation
    # ========================================================

    safe_candidates = [

        result

        for result
        in results

        if (
            result[
                "null_pass_rate"
            ]
            <=
            0.10

            and

            (
                result[
                    "low_context_acceptance_rate"
                ]
                is None

                or

                result[
                    "low_context_acceptance_rate"
                ]
                <=
                0.02
            )
        )
    ]


    print(
        "\n"
        +
        "=" * 118
    )

    print(
        "CANDIDATES WITH:"
    )

    print(
        "null_pass_rate <= 0.10"
    )

    print(
        "low_context_acceptance_rate <= 0.02"
    )

    print(
        "=" * 118
    )


    if not safe_candidates:

        print(
            "No threshold satisfies the safety constraints."
        )

        return


    # Maximize answerable acceptance.
    #
    # On ties:
    # 1) prefer lower complete-context rejection
    # 2) prefer lower null false acceptance
    best = sorted(

        safe_candidates,

        key=lambda result: (
            -result[
                "answerable_pass_rate"
            ],

            result[
                "complete_context_rejection_rate"
            ],

            result[
                "null_pass_rate"
            ],
        )

    )[0]


    for result in safe_candidates:

        print(
            (
                f"threshold={result['threshold']:.2f} "
                f"answerable={result['answerable_pass_rate']:.4f} "
                f"null={result['null_pass_rate']:.4f} "
                f"complete_reject="
                f"{result['complete_context_rejection_rate']:.4f}"
            )
        )


    print(
        "\nBEST UNDER CURRENT CONSTRAINTS:"
    )

    print(
        json.dumps(
            best,
            indent=2,
        )
    )


if __name__ == "__main__":

    main()