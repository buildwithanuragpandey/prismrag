import argparse
import json
import re

from collections import Counter
from pathlib import Path


# ============================================================
# Text normalization
# ============================================================

def normalize_answer(
    text: str | None
) -> str:

    if not text:
        return ""


    text = text.lower()


    text = re.sub(
        r"\[\d+\]",
        " ",
        text
    )


    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


def tokenize(
    text: str | None
) -> list[str]:

    normalized = normalize_answer(
        text
    )

    if not normalized:
        return []

    return normalized.split()


# ============================================================
# Answer metrics
# ============================================================

def exact_match(
    prediction: str | None,
    gold: str | None
) -> float:

    return float(
        normalize_answer(
            prediction
        )
        ==
        normalize_answer(
            gold
        )
    )


def gold_contained_in_answer(
    prediction: str | None,
    gold: str | None
) -> float:

    pred = normalize_answer(
        prediction
    )

    target = normalize_answer(
        gold
    )


    if not pred or not target:
        return 0.0


    return float(
        target in pred
    )


def token_f1(
    prediction: str | None,
    gold: str | None
) -> float:

    pred_tokens = tokenize(
        prediction
    )

    gold_tokens = tokenize(
        gold
    )


    if not pred_tokens or not gold_tokens:
        return 0.0


    pred_counts = Counter(
        pred_tokens
    )

    gold_counts = Counter(
        gold_tokens
    )


    common = (
        pred_counts
        &
        gold_counts
    )


    overlap = sum(
        common.values()
    )


    if overlap == 0:
        return 0.0


    precision = (
        overlap
        /
        len(
            pred_tokens
        )
    )


    recall = (
        overlap
        /
        len(
            gold_tokens
        )
    )


    return (
        2
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )
    )


# ============================================================
# Citation evaluation
# ============================================================

def citation_metrics(
    record: dict
):

    answer = (
        record.get(
            "final_answer"
        )
        or ""
    )


    cited_ids = {

        int(value)

        for value
        in re.findall(
            r"\[(\d+)\]",
            answer
        )
    }


    citation_map = {

        item.get(
            "citation_id"
        ):
            item.get(
                "document_id"
            )

        for item
        in record.get(
            "context_citations",
            []
        )

        if (
            item.get(
                "citation_id"
            )
            is not None
        )
    }


    cited_documents = {

        citation_map[
            citation_id
        ]

        for citation_id
        in cited_ids

        if (
            citation_id
            in citation_map
            and
            citation_map[
                citation_id
            ]
        )
    }


    gold_documents = set(
        record.get(
            "gold_evidence_document_ids",
            []
        )
    )


    if not cited_documents:

        precision = None

    else:

        precision = (
            len(
                cited_documents
                &
                gold_documents
            )
            /
            len(
                cited_documents
            )
        )


    if not gold_documents:

        recall = None

    else:

        recall = (
            len(
                cited_documents
                &
                gold_documents
            )
            /
            len(
                gold_documents
            )
        )


    return {

        "cited_ids":
            sorted(
                cited_ids
            ),

        "cited_document_ids":
            sorted(
                cited_documents
            ),

        "gold_document_precision":
            precision,

        "gold_document_recall":
            recall
    }


# ============================================================
# Aggregate helpers
# ============================================================

def safe_mean(
    values
):

    values = [
        value
        for value in values
        if value is not None
    ]


    if not values:
        return None


    return (
        sum(values)
        /
        len(values)
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    records: list[dict]
):

    evaluated = []


    for record in records:

        if record.get(
            "execution_error"
        ):
            continue


        item = dict(
            record
        )


        if (
            record[
                "is_answerable"
            ]
        ):

            if record[
                "abstained"
            ]:

                item[
                    "exact_match"
                ] = 0.0

                item[
                    "gold_contained"
                ] = 0.0

                item[
                    "token_f1"
                ] = 0.0

            else:

                item[
                    "exact_match"
                ] = exact_match(
                    record.get(
                        "final_answer"
                    ),
                    record.get(
                        "gold_answer"
                    )
                )


                item[
                    "gold_contained"
                ] = (
                    gold_contained_in_answer(
                        record.get(
                            "final_answer"
                        ),
                        record.get(
                            "gold_answer"
                        )
                    )
                )


                item[
                    "token_f1"
                ] = token_f1(
                    record.get(
                        "final_answer"
                    ),
                    record.get(
                        "gold_answer"
                    )
                )


            item[
                "citation_evaluation"
            ] = citation_metrics(
                record
            )


        evaluated.append(
            item
        )


    answerable = [

        item

        for item in evaluated

        if item[
            "is_answerable"
        ]
    ]


    answered = [

        item

        for item in answerable

        if not item[
            "abstained"
        ]
    ]


    null_examples = [

        item

        for item in evaluated

        if not item[
            "is_answerable"
        ]
    ]


    abstention_stage_counts = Counter(

        item.get(
            "abstention_stage"
        )

        for item in answerable

        if item[
            "abstained"
        ]
    )


    summary = {

        "total":
            len(
                evaluated
            ),

        "answerable":
            len(
                answerable
            ),

        "answered_answerable":
            len(
                answered
            ),

        "null_examples":
            len(
                null_examples
            ),


        # ----------------------------------------------------
        # Gold answer metrics
        # ----------------------------------------------------

        "exact_match":
            safe_mean(
                item[
                    "exact_match"
                ]
                for item in answerable
            ),

        "gold_answer_containment":
            safe_mean(
                item[
                    "gold_contained"
                ]
                for item in answerable
            ),

        "token_f1":
            safe_mean(
                item[
                    "token_f1"
                ]
                for item in answerable
            ),


        # ----------------------------------------------------
        # Only among questions actually answered
        # ----------------------------------------------------

        "answered_only_gold_containment":
            safe_mean(
                item[
                    "gold_contained"
                ]
                for item in answered
            ),

        "answered_only_token_f1":
            safe_mean(
                item[
                    "token_f1"
                ]
                for item in answered
            ),


        # ----------------------------------------------------
        # Citation → gold-document alignment
        # ----------------------------------------------------

        "citation_gold_document_precision":
            safe_mean(
                item[
                    "citation_evaluation"
                ][
                    "gold_document_precision"
                ]

                for item in answered
            ),

        "citation_gold_document_recall":
            safe_mean(
                item[
                    "citation_evaluation"
                ][
                    "gold_document_recall"
                ]

                for item in answered
            ),


        "answerable_abstention_stages":
            dict(
                abstention_stage_counts
            )
    }


    return (
        summary,
        evaluated
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = (
        argparse.ArgumentParser()
    )


    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )


    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )


    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()


    with open(
        args.input,
        "r",
        encoding="utf-8"
    ) as file:

        benchmark = json.load(
            file
        )


    summary, records = evaluate(
        benchmark[
            "records"
        ]
    )


    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "source":
                    str(
                        args.input
                    ),

                "summary":
                    summary,

                "records":
                    records
            },
            file,
            ensure_ascii=False,
            indent=2
        )


    print(
        "\n===== OFFLINE ANSWER EVALUATION ====="
    )


    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )


    print(
        "\nSaved:",
        args.output
    )


if __name__ == "__main__":
    main()