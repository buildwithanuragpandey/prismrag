import json

from pathlib import Path


INPUT_PATH = Path(
    "evaluation/results/"
    "e2e_smoke_retrieval_gate_diagnostic.json"
)


def main():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(
            file
        )


    records = (
        payload[
            "records"
        ]
    )


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "ANSWERABLE FALSE REJECTIONS"
    )

    print(
        "=" * 100
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


        if (
            record[
                "initial_evidence_sufficient"
            ]
        ):

            continue


        print(
            "\n"
            +
            "-" * 100
        )


        print(
            "ID:",
            record[
                "id"
            ]
        )


        print(
            "Question:"
        )

        print(
            record[
                "question"
            ]
        )


        print(
            "\nGold docs:"
        )

        print(
            record[
                "gold_document_ids"
            ]
        )


        print(
            "\nRetrieval gold recall:"
        )

        print(
            record[
                "initial_retrieval_gold_recall"
            ]
        )


        print(
            "Context gold recall:"
        )

        print(
            record[
                "initial_context_gold_recall"
            ]
        )


        print(
            "Context complete:"
        )

        print(
            record[
                "initial_context_gold_complete"
            ]
        )


        print(
            "\nEvidence score:"
        )

        print(
            record[
                "initial_evidence_score"
            ]
        )


        print(
            "\nEvidence reasons:"
        )


        for reason in (
            record[
                "initial_evidence_reasons"
            ]
        ):

            print(
                "  -",
                reason,
            )


        if (
            record[
                "rewrite_attempted"
            ]
        ):

            print(
                "\nRewrite:"
            )

            print(
                record[
                    "rewritten_query"
                ]
            )


            print(
                "\nFinal evidence score:"
            )

            print(
                record[
                    "final_evidence_score"
                ]
            )


            print(
                "\nFinal reasons:"
            )


            for reason in (
                record[
                    "final_evidence_reasons"
                ]
            ):

                print(
                    "  -",
                    reason,
                )


    # ========================================================
    # Null false accepts
    # ========================================================

    print(
        "\n\n"
        +
        "=" * 100
    )

    print(
        "NULL FALSE ACCEPTS"
    )

    print(
        "=" * 100
    )


    for record in records:

        if (
            record[
                "question_type"
            ]
            !=
            "null_query"
        ):

            continue


        if not (
            record[
                "initial_evidence_sufficient"
            ]
        ):

            continue


        print(
            "\n"
            +
            "-" * 100
        )


        print(
            "ID:",
            record[
                "id"
            ]
        )


        print(
            "Question:"
        )

        print(
            record[
                "question"
            ]
        )


        print(
            "\nEvidence score:"
        )

        print(
            record[
                "initial_evidence_score"
            ]
        )


        print(
            "\nEvidence reasons:"
        )


        for reason in (
            record[
                "initial_evidence_reasons"
            ]
        ):

            print(
                "  -",
                reason,
            )


    # ========================================================
    # Score overlap
    # ========================================================

    false_reject_scores = [
        record[
            "initial_evidence_score"
        ]

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
                "initial_evidence_sufficient"
            ]
            and
            record[
                "initial_context_gold_recall"
            ]
            is not None
            and
            record[
                "initial_context_gold_recall"
            ]
            >=
            0.75
        )
    ]


    false_accept_scores = [
        record[
            "initial_evidence_score"
        ]

        for record
        in records

        if (
            record[
                "question_type"
            ]
            ==
            "null_query"
            and
            record[
                "initial_evidence_sufficient"
            ]
        )
    ]


    print(
        "\n\n"
        +
        "=" * 100
    )

    print(
        "THRESHOLD SEPARABILITY"
    )

    print(
        "=" * 100
    )


    print(
        "High-context false reject scores:"
    )

    print(
        false_reject_scores
    )


    print(
        "\nNull false accept scores:"
    )

    print(
        false_accept_scores
    )


    if (
        false_reject_scores
        and
        false_accept_scores
    ):

        highest_false_reject = max(
            false_reject_scores
        )

        lowest_false_accept = min(
            false_accept_scores
        )


        print(
            "\nHighest false-reject score:",
            highest_false_reject,
        )


        print(
            "Lowest false-accept score:",
            lowest_false_accept,
        )


        if (
            highest_false_reject
            <
            lowest_false_accept
        ):

            print(
                "\nResult:"
            )

            print(
                (
                    "A scalar threshold alone cannot "
                    "recover all high-quality answerable "
                    "contexts while rejecting this null "
                    "false accept."
                )
            )


if __name__ == "__main__":

    main()