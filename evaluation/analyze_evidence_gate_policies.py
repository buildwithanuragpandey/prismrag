import json
from collections import Counter
from pathlib import Path


INPUT_PATH = Path(
    "evaluation/results/"
    "evidence_gate_v2_500.json"
)


# ============================================================
# Reason classification
# ============================================================

def reason_category(
    reason: str,
) -> str:

    text = (
        reason
        or ""
    ).strip()

    lower = text.lower()


    if text.startswith(
        "Query term coverage is too low:"
    ):
        return "term"


    if text.startswith(
        "Critical query anchors are missing"
    ):
        return "anchor"


    if text.startswith(
        "Potentially weak evidence citations:"
    ):
        return "warning"


    if text.startswith(
        "Evidence appears sufficient."
    ):
        return "info"


    # The candidate policy handles evidence-score
    # thresholds directly, so score-related reasons
    # should not become a second hard blocker.
    if (
        "evidence score" in lower
        and
        (
            "low" in lower
            or
            "below" in lower
            or
            "insufficient" in lower
        )
    ):
        return "score"


    # Everything unknown remains conservative.
    #
    # Examples could include:
    # - too few documents
    # - too few chunks
    # - empty context
    # - another structural safety condition
    return "structural"


# ============================================================
# Candidate decision policy
# ============================================================

def candidate_pass(
    final: dict,
    *,
    score_threshold: float,
    ignore_term_gate: bool,
    ignore_anchor_gate: bool,
) -> bool:

    score = float(
        final.get(
            "evidence_score",
            0.0,
        )
    )


    # Main score requirement stays hard.
    if score < score_threshold:
        return False


    categories = Counter(

        reason_category(
            reason
        )

        for reason
        in (
            final.get(
                "evidence_reasons",
                []
            )
            or []
        )
    )


    # Structural conditions remain hard.
    if categories[
        "structural"
    ]:

        return False


    if (
        not ignore_term_gate
        and
        categories[
            "term"
        ]
    ):

        return False


    if (
        not ignore_anchor_gate
        and
        categories[
            "anchor"
        ]
    ):

        return False


    return True


# ============================================================
# Metrics
# ============================================================

def evaluate_policy(
    records: list[dict],
    *,
    name: str,
    score_threshold: float | None = None,
    ignore_term_gate: bool = False,
    ignore_anchor_gate: bool = False,
    use_current: bool = False,
) -> dict:

    rows = []


    for record in records:

        final = record[
            "final"
        ]


        if use_current:

            passed = bool(
                final[
                    "evidence_sufficient"
                ]
            )

        else:

            passed = candidate_pass(
                final,
                score_threshold=
                    score_threshold,
                ignore_term_gate=
                    ignore_term_gate,
                ignore_anchor_gate=
                    ignore_anchor_gate,
            )


        rows.append(
            {
                "record":
                    record,

                "passed":
                    passed,
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


    current_answerable_pass = {
        row[
            "record"
        ][
            "id"
        ]

        for row
        in answerable

        if row[
            "record"
        ][
            "final"
        ][
            "evidence_sufficient"
        ]
    }


    current_null_pass = {
        row[
            "record"
        ][
            "id"
        ]

        for row
        in null_examples

        if row[
            "record"
        ][
            "final"
        ][
            "evidence_sufficient"
        ]
    }


    new_answerable_pass = {
        row[
            "record"
        ][
            "id"
        ]

        for row
        in answerable

        if row[
            "passed"
        ]
    }


    new_null_pass = {
        row[
            "record"
        ][
            "id"
        ]

        for row
        in null_examples

        if row[
            "passed"
        ]
    }


    # ========================================================
    # Complete context
    # ========================================================

    complete_context = [
        row
        for row
        in answerable
        if (
            row[
                "record"
            ][
                "final"
            ][
                "context_gold_recall"
            ]
            ==
            1.0
        )
    ]


    complete_rejected = [
        row
        for row
        in complete_context
        if not row[
            "passed"
        ]
    ]


    # ========================================================
    # High context recall
    # ========================================================

    high_context = [
        row
        for row
        in answerable
        if (
            row[
                "record"
            ][
                "final"
            ][
                "context_gold_recall"
            ]
            is not None
            and
            row[
                "record"
            ][
                "final"
            ][
                "context_gold_recall"
            ]
            >=
            0.75
        )
    ]


    high_rejected = [
        row
        for row
        in high_context
        if not row[
            "passed"
        ]
    ]


    # ========================================================
    # Low context recall
    # ========================================================

    low_context = [
        row
        for row
        in answerable
        if (
            row[
                "record"
            ][
                "final"
            ][
                "context_gold_recall"
            ]
            is not None
            and
            row[
                "record"
            ][
                "final"
            ][
                "context_gold_recall"
            ]
            <=
            0.25
        )
    ]


    low_accepted = [
        row
        for row
        in low_context
        if row[
            "passed"
        ]
    ]


    return {
        "name":
            name,

        "answerable_sufficient_rate":
            (
                len(
                    new_answerable_pass
                )
                /
                len(
                    answerable
                )
            ),

        "null_sufficient_rate":
            (
                len(
                    new_null_pass
                )
                /
                len(
                    null_examples
                )
            ),

        "null_correct_rejection_rate":
            (
                1.0
                -
                (
                    len(
                        new_null_pass
                    )
                    /
                    len(
                        null_examples
                    )
                )
            ),

        "complete_context_rejection_rate":
            (
                len(
                    complete_rejected
                )
                /
                len(
                    complete_context
                )
            ),

        "high_context_rejection_rate":
            (
                len(
                    high_rejected
                )
                /
                len(
                    high_context
                )
            ),

        "low_context_acceptance_rate":
            (
                len(
                    low_accepted
                )
                /
                len(
                    low_context
                )

                if low_context
                else 0.0
            ),

        # Newly rescued answerable cases
        "answerable_rescued_vs_current":
            len(
                new_answerable_pass
                -
                current_answerable_pass
            ),

        # Regressions where a previously accepted
        # answerable example becomes rejected.
        "answerable_lost_vs_current":
            len(
                current_answerable_pass
                -
                new_answerable_pass
            ),

        # New safety failures among null queries.
        "new_null_false_accepts_vs_current":
            len(
                new_null_pass
                -
                current_null_pass
            ),

        # Null queries that candidate policy fixes.
        "null_false_accepts_fixed_vs_current":
            len(
                current_null_pass
                -
                new_null_pass
            ),
    }


# ============================================================
# Reason statistics
# ============================================================

def reason_statistics(
    records,
):

    overall = Counter()

    rejected_answerable = Counter()

    rejected_null = Counter()


    for record in records:

        final = record[
            "final"
        ]


        categories = {

            reason_category(
                reason
            )

            for reason
            in (
                final.get(
                    "evidence_reasons",
                    []
                )
                or []
            )
        }


        for category in categories:

            overall[
                category
            ] += 1


        if not final[
            "evidence_sufficient"
        ]:

            target = (
                rejected_answerable
                if record[
                    "is_answerable"
                ]
                else
                rejected_null
            )


            for category in categories:

                target[
                    category
                ] += 1


    return {
        "overall":
            dict(
                overall
            ),

        "rejected_answerable":
            dict(
                rejected_answerable
            ),

        "rejected_null":
            dict(
                rejected_null
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


    print(
        "=" * 100
    )

    print(
        "EVIDENCE GATE POLICY ANALYSIS"
    )

    print(
        "=" * 100
    )


    print(
        "\nReason statistics:"
    )

    print(
        json.dumps(
            reason_statistics(
                records
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


    # ========================================================
    # Policies
    # ========================================================

    policies = [

        {
            "name":
                "CURRENT",

            "use_current":
                True,
        },


        # ----------------------------------------------------
        # Minimal change:
        #
        # lexical term coverage no longer has veto power.
        # Critical anchors are still hard.
        # ----------------------------------------------------

        {
            "name":
                "TERM_SOFT_SCORE_070",

            "score_threshold":
                0.70,

            "ignore_term_gate":
                True,

            "ignore_anchor_gate":
                False,
        },


        # Same policy but slightly stricter global score.
        {
            "name":
                "TERM_SOFT_SCORE_075",

            "score_threshold":
                0.75,

            "ignore_term_gate":
                True,

            "ignore_anchor_gate":
                False,
        },


        # ----------------------------------------------------
        # Upper-bound diagnostic:
        #
        # Neither lexical coverage nor current anchor
        # extraction can veto the answer.
        #
        # Structural blockers remain hard.
        # ----------------------------------------------------

        {
            "name":
                "LEXICAL_SOFT_SCORE_070",

            "score_threshold":
                0.70,

            "ignore_term_gate":
                True,

            "ignore_anchor_gate":
                True,
        },


        {
            "name":
                "LEXICAL_SOFT_SCORE_075",

            "score_threshold":
                0.75,

            "ignore_term_gate":
                True,

            "ignore_anchor_gate":
                True,
        },
    ]


    results = []


    for policy in policies:

        result = evaluate_policy(
            records,
            **policy,
        )

        results.append(
            result
        )


    # ========================================================
    # Table
    # ========================================================

    print(
        "\n"
        +
        "=" * 130
    )

    print(
        "POLICY COMPARISON"
    )

    print(
        "=" * 130
    )


    header = (
        f"{'POLICY':28s}"
        f"{'ANS PASS':>10s}"
        f"{'NULL PASS':>11s}"
        f"{'COMP REJ':>11s}"
        f"{'HIGH REJ':>11s}"
        f"{'LOW ACC':>10s}"
        f"{'RESCUE':>9s}"
        f"{'ANS LOST':>10s}"
        f"{'NEW NULL':>10s}"
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
            f"{result['name']:28s}"

            f"{result['answerable_sufficient_rate']:10.4f}"

            f"{result['null_sufficient_rate']:11.4f}"

            f"{result['complete_context_rejection_rate']:11.4f}"

            f"{result['high_context_rejection_rate']:11.4f}"

            f"{result['low_context_acceptance_rate']:10.4f}"

            f"{result['answerable_rescued_vs_current']:9d}"

            f"{result['answerable_lost_vs_current']:10d}"

            f"{result['new_null_false_accepts_vs_current']:10d}"
        )


    print(
        "\nLegend:"
    )

    print(
        "ANS PASS = answerable sufficient rate"
    )

    print(
        "NULL PASS = null false-accept rate (lower is better)"
    )

    print(
        "COMP REJ = complete-context rejection rate"
    )

    print(
        "HIGH REJ = context recall >= 0.75 rejection rate"
    )

    print(
        "LOW ACC = context recall <= 0.25 acceptance rate"
    )

    print(
        "RESCUE = newly accepted answerable cases"
    )

    print(
        "ANS LOST = previously accepted answerable cases lost"
    )

    print(
        "NEW NULL = additional null false accepts"
    )


if __name__ == "__main__":

    main()