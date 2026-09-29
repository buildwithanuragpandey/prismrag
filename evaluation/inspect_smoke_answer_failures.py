import json
from pathlib import Path


INPUT_PATH = Path(
    "evaluation/results/end_to_end/"
    "smoke_10_offline_eval_v2.json"
)


def print_value(
    label,
    value,
):
    if value is None:
        value = "-"

    print(
        f"{label}: {value}"
    )


def main():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(file)


    records = payload.get(
        "records",
        [],
    )


    answerable = [
        record
        for record in records
        if (
            record.get(
                "is_answerable"
            )
            and
            not record.get(
                "execution_error"
            )
        )
    ]


    print(
        "=" * 110
    )

    print(
        "ANSWERABLE SMOKE AUDIT"
    )

    print(
        "=" * 110
    )

    print(
        "Answerable:",
        len(answerable),
    )


    # ========================================================
    # First answered examples
    # ========================================================

    answered = [
        record
        for record in answerable
        if not record.get(
            "abstained"
        )
    ]


    print(
        "\n"
        +
        "=" * 110
    )

    print(
        "A) GENERATED ANSWERS"
    )

    print(
        "=" * 110
    )


    for record in answered:

        evaluation = (
            record.get(
                "offline_evaluation",
                {}
            )
            or {}
        )


        print(
            "\n"
            +
            "-" * 110
        )

        print(
            "ID:",
            record.get("id"),
        )

        print_value(
            "Question type",
            record.get(
                "gold_question_type"
            ),
        )

        print(
            "\nQUESTION:"
        )

        print(
            record.get(
                "question"
            )
        )


        print(
            "\nGOLD ANSWER:"
        )

        print(
            record.get(
                "gold_answer"
            )
        )


        print(
            "\nFINAL ANSWER:"
        )

        print(
            record.get(
                "final_answer"
            )
        )


        raw_answer = (
            record.get(
                "raw_answer"
            )
        )

        if raw_answer:

            print(
                "\nRAW ANSWER:"
            )

            print(
                raw_answer
            )


        print(
            "\n--- PIPELINE ---"
        )

        print_value(
            "Evidence sufficient",
            record.get(
                "evidence_sufficient"
            ),
        )

        print_value(
            "Rewritten",
            record.get(
                "rewritten"
            ),
        )

        print_value(
            "Retry count",
            record.get(
                "retry_count"
            ),
        )

        print_value(
            "Supported claims",
            record.get(
                "supported_claims"
            ),
        )

        print_value(
            "Unsupported claims",
            record.get(
                "unsupported_claims"
            ),
        )

        print_value(
            "Relevant claims",
            record.get(
                "relevant_claims"
            ),
        )

        print_value(
            "Filtered irrelevant claims",
            record.get(
                "filtered_irrelevant_claims"
            ),
        )

        print_value(
            "Citation valid",
            record.get(
                "citation_valid"
            ),
        )

        print_value(
            "Internal answer passed",
            record.get(
                "answer_passed"
            ),
        )


        print(
            "\n--- OFFLINE EVALUATOR V2 ---"
        )

        print_value(
            "Answer type",
            evaluation.get(
                "answer_type"
            ),
        )

        print_value(
            "Correct",
            evaluation.get(
                "answer_correct"
            ),
        )

        print_value(
            "Method",
            evaluation.get(
                "method"
            ),
        )

        print_value(
            "Confidence",
            evaluation.get(
                "confidence"
            ),
        )

        print_value(
            "Exact match",
            evaluation.get(
                "exact_match"
            ),
        )

        print_value(
            "Gold contained",
            evaluation.get(
                "gold_contained"
            ),
        )

        print_value(
            "Token F1",
            evaluation.get(
                "token_f1"
            ),
        )


        nli = evaluation.get(
            "nli"
        )

        if nli:

            print(
                "NLI:",
                nli,
            )


        citation = (
            evaluation.get(
                "citation",
                {}
            )
            or {}
        )


        print(
            "\n--- CITATIONS ---"
        )

        print_value(
            "Gold evidence docs",
            record.get(
                "gold_evidence_document_ids"
            ),
        )

        print_value(
            "Cited docs",
            citation.get(
                "cited_document_ids"
            ),
        )

        print_value(
            "Dataset citation precision",
            citation.get(
                "dataset_evidence_document_precision"
            ),
        )

        print_value(
            "Dataset citation recall",
            citation.get(
                "dataset_evidence_document_recall"
            ),
        )


    # ========================================================
    # Abstention cases
    # ========================================================

    abstained = [
        record
        for record in answerable
        if record.get(
            "abstained"
        )
    ]


    print(
        "\n\n"
        +
        "=" * 110
    )

    print(
        "B) FALSE ABSTENTIONS"
    )

    print(
        "=" * 110
    )


    for record in abstained:

        evaluation = (
            record.get(
                "offline_evaluation",
                {}
            )
            or {}
        )


        print(
            "\n"
            +
            "-" * 110
        )

        print(
            "ID:",
            record.get("id"),
        )


        print(
            "\nQUESTION:"
        )

        print(
            record.get(
                "question"
            )
        )


        print(
            "\nGOLD ANSWER:"
        )

        print(
            record.get(
                "gold_answer"
            )
        )


        print(
            "\n--- FAILURE LOCATION ---"
        )

        print_value(
            "Abstention stage",
            evaluation.get(
                "abstention_stage"
            ),
        )

        print_value(
            "Evidence sufficient",
            record.get(
                "evidence_sufficient"
            ),
        )

        print_value(
            "Rewritten",
            record.get(
                "rewritten"
            ),
        )

        print_value(
            "Retry count",
            record.get(
                "retry_count"
            ),
        )

        print_value(
            "Retrieved gold recall",
            record.get(
                "retrieved_gold_document_recall"
            ),
        )

        print_value(
            "Context gold recall",
            record.get(
                "context_gold_document_recall"
            ),
        )

        print_value(
            "Supported claims",
            record.get(
                "supported_claims"
            ),
        )

        print_value(
            "Unsupported claims",
            record.get(
                "unsupported_claims"
            ),
        )


        print(
            "\nGold evidence docs:"
        )

        print(
            record.get(
                "gold_evidence_document_ids"
            )
        )


        print(
            "\nRetrieved document ids:"
        )

        print(
            record.get(
                "retrieved_document_ids"
            )
        )


        print(
            "\nContext document ids:"
        )

        print(
            record.get(
                "context_document_ids"
            )
        )


if __name__ == "__main__":
    main()