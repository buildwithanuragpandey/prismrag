import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from sentence_transformers import CrossEncoder


NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

YES = {
    "yes",
    "true",
}

NO = {
    "no",
    "false",
}

MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}


# ============================================================
# Normalization
# ============================================================

def strip_citations(
    text
):
    return re.sub(
        r"\[\s*\d+\s*\]",
        " ",
        text or ""
    )


def normalize(
    text
):
    text = unicodedata.normalize(
        "NFKC",
        strip_citations(
            text
        )
    )

    text = text.lower()

    text = text.replace(
        "’",
        "'"
    )

    text = re.sub(
        r"[^a-z0-9%.$€£'\-\s]",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def word_tokens(
    text
):
    return re.findall(
        r"[a-z0-9]+",
        normalize(
            text
        )
    )


# ============================================================
# Legacy token F1
# ============================================================

def token_f1(
    prediction,
    gold
):
    pred = word_tokens(
        prediction
    )

    ref = word_tokens(
        gold
    )

    if not pred and not ref:
        return 1.0

    if not pred or not ref:
        return 0.0

    pred_counts = Counter(
        pred
    )

    ref_counts = Counter(
        ref
    )

    overlap = sum(
        min(
            pred_counts[token],
            ref_counts[token]
        )
        for token
        in (
            pred_counts.keys()
            &
            ref_counts.keys()
        )
    )

    if not overlap:
        return 0.0

    precision = (
        overlap
        /
        len(
            pred
        )
    )

    recall = (
        overlap
        /
        len(
            ref
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
# Answer type detection
# ============================================================

def looks_like_date(
    text
):
    normalized = normalize(
        text
    )

    tokens = word_tokens(
        text
    )

    if (
        not normalized
        or
        len(
            tokens
        ) > 8
    ):
        return False

    if (
        any(
            token in MONTHS
            for token
            in tokens
        )
        and
        re.search(
            r"\b\d{1,4}\b",
            normalized
        )
    ):
        return True

    if re.fullmatch(
        r"(?:19|20)\d{2}",
        normalized
    ):
        return True

    return bool(
        re.fullmatch(
            (
                r"\d{1,2}"
                r"[\/\-]"
                r"\d{1,2}"
                r"[\/\-]"
                r"(?:\d{2}|\d{4})"
            ),
            normalized
        )
    )


def looks_like_number(
    text
):
    return bool(
        re.fullmatch(
            (
                r"[$€£]?\s*"
                r"-?\d[\d,]*"
                r"(?:\.\d+)?"
                r"(?:\s*"
                r"(?:"
                r"%|percent|percentage|"
                r"million|billion|thousand|"
                r"km|kg|"
                r"miles?|hours?|minutes?|"
                r"seconds?|years?|days?"
                r")"
                r")?"
            ),
            normalize(
                text
            )
        )
    )


def answer_type(
    gold,
    is_answerable
):
    if not is_answerable:
        return "null"

    normalized = normalize(
        gold
    )

    if normalized in (
        YES
        |
        NO
    ):
        return "boolean"

    if looks_like_date(
        gold or ""
    ):
        return "date"

    if looks_like_number(
        gold or ""
    ):
        return "numeric"

    if len(
        word_tokens(
            gold
        )
    ) <= 8:
        return "short_span"

    return "free_form"


# ============================================================
# Local NLI scorer
# ============================================================

class NLIScorer:

    def __init__(
        self
    ):
        self.model = None


    def score(
        self,
        premise,
        hypothesis
    ):
        if self.model is None:

            print(
                (
                    "Loading offline NLI evaluator: "
                    f"{NLI_MODEL}"
                )
            )

            self.model = CrossEncoder(
                NLI_MODEL
            )


        logits = np.asarray(

            self.model.predict(
                [
                    (
                        premise,
                        hypothesis
                    )
                ],
                show_progress_bar=False
            ),

            dtype=float
        )


        if logits.ndim == 2:

            logits = logits[
                0
            ]


        # Stable softmax

        logits = (
            logits
            -
            logits.max()
        )

        probabilities = np.exp(
            logits
        )

        probabilities /= (
            probabilities.sum()
        )


        labels = [
            "contradiction",
            "entailment",
            "neutral",
        ]


        best_index = int(
            np.argmax(
                probabilities
            )
        )


        return {

            "label":
                labels[
                    best_index
                ],

            "confidence":
                float(
                    probabilities[
                        best_index
                    ]
                ),

            **{
                label:
                    float(
                        probabilities[
                            index
                        ]
                    )

                for index, label
                in enumerate(
                    labels
                )
            }
        }


# ============================================================
# Boolean handling
# ============================================================

def explicit_boolean(
    answer
):
    text = normalize(
        answer
    )

    prefix = " ".join(
        text.split()[
            :8
        ]
    )


    if re.match(
        r"^(yes|true)\b",
        prefix
    ):
        return "yes"


    if re.match(
        r"^(no|false)\b",
        prefix
    ):
        return "no"


    negative_patterns = (

        r"\bthere (?:was|is|were|are) no\b",

        r"\bthere (?:was|is|were|are) not\b",

        r"\bdid not\b",

        r"\bdoes not\b",

        r"\bdo not\b",

        r"\bwas not\b",

        r"\bwere not\b",

        r"\bis not\b",

        r"\bare not\b",

        r"\bhas not\b",

        r"\bhave not\b",

        r"\bno change\b",

        r"\bno difference\b",
    )


    if any(
        re.search(
            pattern,
            text
        )
        for pattern
        in negative_patterns
    ):
        return "no"


    positive_patterns = (

        r"\bthere was "
        r"(?:a|an|indeed|significant|clear|notable)\b",

        r"\bthere is "
        r"(?:a|an|indeed|significant|clear|notable)\b",

        r"\ba change\b",

        r"\ba difference\b",

        r"\bsignificant shift\b",

        r"\bclear shift\b",
    )


    if any(
        re.search(
            pattern,
            text
        )
        for pattern
        in positive_patterns
    ):
        return "yes"


    return None


def positive_proposition(
    question
):
    question = re.sub(
        r"\s+",
        " ",
        (
            question
            or ""
        ).strip()
    ).rstrip(
        " ?"
    )


    patterns = [

        (
            r"^was there (.+)$",
            r"There was \1."
        ),

        (
            r"^is there (.+)$",
            r"There is \1."
        ),

        (
            r"^were there (.+)$",
            r"There were \1."
        ),

        (
            r"^are there (.+)$",
            r"There are \1."
        ),

        (
            r"^has there been (.+)$",
            r"There has been \1."
        ),

        (
            r"^have there been (.+)$",
            r"There have been \1."
        ),

        (
            r"^does (.+)$",
            r"\1."
        ),

        (
            r"^did (.+)$",
            r"\1."
        ),
    ]


    for pattern, replacement in (
        patterns
    ):

        if re.match(
            pattern,
            question,
            flags=re.IGNORECASE
        ):

            return re.sub(
                pattern,
                replacement,
                question,
                count=1,
                flags=re.IGNORECASE
            )


    return None


# ============================================================
# Numeric / date helpers
# ============================================================

def numbers(
    text
):
    return [

        number.replace(
            ",",
            ""
        )

        for number
        in re.findall(
            r"-?\d[\d,]*(?:\.\d+)?",
            normalize(
                text
            )
        )
    ]


def number_match(
    prediction,
    gold
):
    reference_numbers = numbers(
        gold
    )

    predicted_numbers = numbers(
        prediction
    )

    return (
        bool(
            reference_numbers
        )
        and
        all(
            number
            in predicted_numbers

            for number
            in reference_numbers
        )
    )


def date_match(
    prediction,
    gold
):
    gold_normalized = normalize(
        gold
    )

    prediction_normalized = normalize(
        prediction
    )


    if (
        gold_normalized
        and
        gold_normalized
        in prediction_normalized
    ):
        return True


    reference_numbers = numbers(
        gold
    )

    predicted_numbers = numbers(
        prediction
    )


    if (
        reference_numbers
        and
        not all(
            number
            in predicted_numbers

            for number
            in reference_numbers
        )
    ):
        return False


    reference_months = {

        token

        for token
        in word_tokens(
            gold
        )

        if token
        in MONTHS
    }


    if (
        reference_months
        and
        not reference_months.issubset(
            set(
                word_tokens(
                    prediction
                )
            )
        )
    ):
        return False


    return bool(
        reference_numbers
        or
        reference_months
    )


# ============================================================
# Answer scorer
# ============================================================

def score_answer(
    record,
    kind,
    nli
):
    gold = (
        record.get(
            "gold_answer"
        )
        or ""
    )

    answer = (
        record.get(
            "final_answer"
        )
        or ""
    )

    abstained = bool(
        record.get(
            "abstained"
        )
    )


    gold_normalized = normalize(
        gold
    )

    answer_normalized = normalize(
        answer
    )


    result = {

        "exact_match":
            float(
                bool(
                    gold_normalized
                )
                and
                gold_normalized
                ==
                answer_normalized
            ),

        "gold_contained":
            float(
                bool(
                    gold_normalized
                )
                and
                gold_normalized
                in
                answer_normalized
            ),

        "token_f1":
            token_f1(
                answer,
                gold
            ),

        "answer_correct":
            None,

        "method":
            None,

        "confidence":
            None,

        "nli":
            None,
    }


    # --------------------------------------------------------
    # Answerable question but pipeline abstained
    # --------------------------------------------------------

    if abstained:

        result.update(

            answer_correct=False,

            method=(
                "answerable_abstention"
            ),

            confidence=1.0
        )

        return result


    # --------------------------------------------------------
    # Boolean
    # --------------------------------------------------------

    if kind == "boolean":

        expected = (
            "yes"
            if gold_normalized
            in YES
            else "no"
        )


        direct = explicit_boolean(
            answer
        )


        if direct is not None:

            result.update(

                answer_correct=(
                    direct
                    ==
                    expected
                ),

                method=(
                    "boolean_explicit_or_polarity"
                ),

                confidence=1.0
            )

            return result


        proposition = (
            positive_proposition(
                record.get(
                    "question"
                )
            )
        )


        if proposition is None:

            result[
                "method"
            ] = (
                "boolean_unresolved"
            )

            return result


        semantic = nli.score(
            answer,
            proposition
        )


        expected_label = (
            "entailment"
            if expected
            ==
            "yes"
            else
            "contradiction"
        )


        result.update(

            answer_correct=(

                semantic[
                    "label"
                ]
                ==
                expected_label

                and

                semantic[
                    "confidence"
                ]
                >=
                0.50
            ),

            method=(
                "boolean_nli"
            ),

            confidence=(
                semantic[
                    "confidence"
                ]
            ),

            nli=semantic,

            boolean_proposition=(
                proposition
            )
        )


        return result


    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    if kind == "numeric":

        result.update(

            answer_correct=(
                number_match(
                    answer,
                    gold
                )
            ),

            method="numeric_match",

            confidence=1.0
        )

        return result


    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if kind == "date":

        result.update(

            answer_correct=(
                date_match(
                    answer,
                    gold
                )
            ),

            method="date_match",

            confidence=1.0
        )

        return result


    # --------------------------------------------------------
    # Entity / short span
    # --------------------------------------------------------

    if kind == "short_span":

        result.update(

            answer_correct=(

                bool(
                    gold_normalized
                )

                and

                gold_normalized
                in
                answer_normalized
            ),

            method=(
                "short_span_containment"
            ),

            confidence=1.0
        )

        return result


    # --------------------------------------------------------
    # Free form
    # --------------------------------------------------------

    if (
        not gold_normalized
        or
        not answer_normalized
    ):

        result[
            "method"
        ] = (
            "free_form_unresolved"
        )

        return result


    semantic = nli.score(
        answer,
        gold
    )


    result.update(

        answer_correct=(

            semantic[
                "label"
            ]
            ==
            "entailment"

            and

            semantic[
                "confidence"
            ]
            >=
            0.50
        ),

        method=(
            "free_form_nli"
        ),

        confidence=(
            semantic[
                "confidence"
            ]
        ),

        nli=semantic
    )


    return result


# ============================================================
# Citation evaluation
# ============================================================

def citation_eval(
    record
):
    gold_documents = set(
        record.get(
            "gold_evidence_document_ids",
            []
        )
        or []
    )


    citation_map = {}


    for item in (
        record.get(
            "context_citations",
            []
        )
        or []
    ):

        citation_id = item.get(
            "citation_id"
        )

        document_id = item.get(
            "document_id"
        )


        if (
            citation_id is not None
            and
            document_id is not None
        ):

            citation_map[
                str(
                    citation_id
                )
            ] = (
                document_id
            )


    citation_ids = list(
        dict.fromkeys(

            re.findall(
                r"\[\s*(\d+)\s*\]",
                record.get(
                    "final_answer"
                )
                or ""
            )
        )
    )


    cited_documents = list(
        dict.fromkeys(

            citation_map[
                citation_id
            ]

            for citation_id
            in citation_ids

            if citation_id
            in citation_map
        )
    )


    cited_set = set(
        cited_documents
    )


    precision = (

        len(
            cited_set
            &
            gold_documents
        )
        /
        len(
            cited_set
        )

        if cited_set

        else None
    )


    recall = (

        len(
            cited_set
            &
            gold_documents
        )
        /
        len(
            gold_documents
        )

        if gold_documents

        else None
    )


    return {

        "citation_ids":
            citation_ids,

        "cited_document_ids":
            cited_documents,

        "unresolved_citation_ids":
            [

                citation_id

                for citation_id
                in citation_ids

                if citation_id
                not in citation_map
            ],

        "dataset_evidence_document_precision":
            precision,

        "dataset_evidence_document_recall":
            recall,
    }


# ============================================================
# Aggregate evaluation
# ============================================================

def safe_mean(
    values
):
    values = [

        value

        for value
        in values

        if value is not None
    ]

    return (
        mean(
            values
        )
        if values
        else None
    )


def evaluate(
    records
):
    nli = NLIScorer()

    output = []


    for record in records:

        if record.get(
            "execution_error"
        ):

            output.append(
                {
                    **record,

                    "offline_evaluation":
                        {
                            "scorable":
                                False,

                            "reason":
                                "execution_error",
                        }
                }
            )

            continue


        is_answerable = bool(
            record.get(
                "is_answerable"
            )
        )

        abstained = bool(
            record.get(
                "abstained"
            )
        )


        kind = answer_type(

            record.get(
                "gold_answer"
            ),

            is_answerable
        )


        if is_answerable:

            correctness = (
                score_answer(
                    record,
                    kind,
                    nli
                )
            )

        else:

            correctness = {

                "answer_correct":
                    abstained,

                "method":
                    "null_abstention",

                "confidence":
                    1.0,

                "exact_match":
                    None,

                "gold_contained":
                    None,

                "token_f1":
                    None,

                "nli":
                    None,
            }


        abstention_stage = None


        if (
            is_answerable
            and
            abstained
        ):

            abstention_stage = (

                "evidence_gate"

                if record.get(
                    "evidence_sufficient"
                )
                is False

                else

                "generation_or_grounding"
            )


        output.append(

            {
                **record,

                "offline_evaluation":
                    {

                        "scorable":
                            (
                                correctness.get(
                                    "answer_correct"
                                )
                                is not None
                            ),

                        "answer_type":
                            kind,

                        "abstention_stage":
                            abstention_stage,

                        **correctness,

                        "citation":
                            citation_eval(
                                record
                            ),
                    }
            }
        )


    successful = [

        record

        for record
        in output

        if not record.get(
            "execution_error"
        )
    ]


    answerable = [

        record

        for record
        in successful

        if record.get(
            "is_answerable"
        )
    ]


    answered = [

        record

        for record
        in answerable

        if not record.get(
            "abstained"
        )
    ]


    null_examples = [

        record

        for record
        in successful

        if not record.get(
            "is_answerable"
        )
    ]


    scorable = [

        record

        for record
        in answered

        if (
            record[
                "offline_evaluation"
            ].get(
                "answer_correct"
            )
            is not None
        )
    ]


    correct_answered = [

        record

        for record
        in scorable

        if (
            record[
                "offline_evaluation"
            ][
                "answer_correct"
            ]
            is True
        )
    ]


    correct_all_answerable = [

        record

        for record
        in answerable

        if (
            record[
                "offline_evaluation"
            ].get(
                "answer_correct"
            )
            is True
        )
    ]


    type_distribution = Counter(

        record[
            "offline_evaluation"
        ][
            "answer_type"
        ]

        for record
        in successful
    )


    by_type = {}


    for kind in sorted(
        type_distribution
    ):

        subset = [

            record

            for record
            in answered

            if (
                record[
                    "offline_evaluation"
                ][
                    "answer_type"
                ]
                ==
                kind
            )

            and

            (
                record[
                    "offline_evaluation"
                ].get(
                    "answer_correct"
                )
                is not None
            )
        ]


        if subset:

            by_type[
                kind
            ] = {

                "count":
                    len(
                        subset
                    ),

                "accuracy":
                    safe_mean(

                        float(
                            record[
                                "offline_evaluation"
                            ][
                                "answer_correct"
                            ]
                        )

                        for record
                        in subset
                    )
            }


    citation_precision = [

        record[
            "offline_evaluation"
        ][
            "citation"
        ][
            "dataset_evidence_document_precision"
        ]

        for record
        in answered

        if (
            record[
                "offline_evaluation"
            ][
                "citation"
            ][
                "dataset_evidence_document_precision"
            ]
            is not None
        )
    ]


    citation_recall = [

        record[
            "offline_evaluation"
        ][
            "citation"
        ][
            "dataset_evidence_document_recall"
        ]

        for record
        in answered

        if (
            record[
                "offline_evaluation"
            ][
                "citation"
            ][
                "dataset_evidence_document_recall"
            ]
            is not None
        )
    ]


    abstention_stages = Counter(

        record[
            "offline_evaluation"
        ].get(
            "abstention_stage"
        )

        for record
        in answerable

        if record.get(
            "abstained"
        )
    )


    abstention_stages.pop(
        None,
        None
    )


    summary = {

        "total":
            len(
                successful
            ),

        "execution_errors":
            (
                len(
                    output
                )
                -
                len(
                    successful
                )
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

        "answer_type_distribution":
            dict(
                type_distribution
            ),


        # ----------------------------------------------------
        # Main correctness metrics
        # ----------------------------------------------------

        "automatic_correctness_coverage":
            (
                len(
                    scorable
                )
                /
                len(
                    answered
                )

                if answered

                else None
            ),

        "answered_only_auto_accuracy":
            (
                len(
                    correct_answered
                )
                /
                len(
                    scorable
                )

                if scorable

                else None
            ),

        # Conservative end-to-end metric.
        # Answerable abstentions count as failures.
        # Unresolved automatic cases are not counted as correct.
        "end_to_end_auto_correct_rate_lower_bound":
            (
                len(
                    correct_all_answerable
                )
                /
                len(
                    answerable
                )

                if answerable

                else None
            ),


        # ----------------------------------------------------
        # Legacy metrics kept for comparison
        # ----------------------------------------------------

        "legacy_exact_match":
            safe_mean(

                record[
                    "offline_evaluation"
                ].get(
                    "exact_match"
                )

                for record
                in answerable
            ),

        "legacy_token_f1":
            safe_mean(

                record[
                    "offline_evaluation"
                ].get(
                    "token_f1"
                )

                for record
                in answerable
            ),


        # ----------------------------------------------------
        # Null behavior
        # ----------------------------------------------------

        "correct_null_abstention_rate":
            safe_mean(

                float(
                    record.get(
                        "abstained",
                        False
                    )
                )

                for record
                in null_examples
            ),


        # ----------------------------------------------------
        # Dataset evidence citation metrics
        # ----------------------------------------------------

        "citation_dataset_evidence_precision":
            safe_mean(
                citation_precision
            ),

        "citation_dataset_evidence_recall":
            safe_mean(
                citation_recall
            ),


        "answerable_abstention_stages":
            dict(
                abstention_stages
            ),

        "accuracy_by_answer_type":
            by_type,
    }


    return {

        "summary":
            summary,

        "records":
            output,
    }


# ============================================================
# Cheap self-test
# ============================================================

def self_test():

    assert (
        answer_type(
            "Yes",
            True
        )
        ==
        "boolean"
    )

    assert (
        answer_type(
            "Valve",
            True
        )
        ==
        "short_span"
    )

    assert (
        answer_type(
            "November 16, 2023",
            True
        )
        ==
        "date"
    )

    assert (
        answer_type(
            "42",
            True
        )
        ==
        "numeric"
    )

    assert (
        answer_type(
            None,
            False
        )
        ==
        "null"
    )

    assert (
        normalize(
            "Valve [1]"
        )
        ==
        "valve"
    )

    assert (
        token_f1(
            "Valve [1]",
            "Valve"
        )
        ==
        1.0
    )

    assert number_match(
        "The answer is 42.",
        "42"
    )

    assert date_match(
        "It happened on November 16, 2023.",
        "November 16, 2023"
    )

    assert (
        positive_proposition(
            "Was there a change in the reporting?"
        )
        ==
        "There was a change in the reporting."
    )

    assert (
        explicit_boolean(
            (
                "There was a significant change "
                "in the reporting. [1]"
            )
        )
        ==
        "yes"
    )

    assert (
        explicit_boolean(
            (
                "There was no change "
                "in the reporting. [1]"
            )
        )
        ==
        "no"
    )

    print(
        "Answer evaluator V2 self-test: OK"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path
    )

    parser.add_argument(
        "--output",
        type=Path
    )

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    parser.add_argument(
        "--details",
        action="store_true"
    )

    args = parser.parse_args()


    if args.self_test:

        self_test()

        return


    if args.input is None:

        raise SystemExit(
            (
                "--input is required unless "
                "--self-test is used"
            )
        )


    with open(
        args.input,
        "r",
        encoding="utf-8"
    ) as file:

        payload = json.load(
            file
        )


    result = evaluate(

        payload.get(
            "records",
            []
        )
    )


    output_path = (

        args.output

        or

        args.input.with_name(
            (
                f"{args.input.stem}"
                "_offline_eval_v2.json"
            )
        )
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "evaluator":
                    "answer_evaluator_v2",

                "source":
                    str(
                        args.input
                    ),

                **result,
            },
            file,
            ensure_ascii=False,
            indent=2
        )


    print(
        "\n"
        "===== OFFLINE ANSWER EVALUATION V2 ====="
    )

    print(
        json.dumps(
            result[
                "summary"
            ],
            ensure_ascii=False,
            indent=2
        )
    )


    if args.details:

        print(
            "\n"
            "===== PER-EXAMPLE ====="
        )


        for record in (
            result[
                "records"
            ]
        ):

            evaluation = (
                record.get(
                    "offline_evaluation",
                    {}
                )
            )


            print(
                record.get(
                    "id"
                ),
                "|",
                evaluation.get(
                    "answer_type"
                ),
                "| correct=",
                evaluation.get(
                    "answer_correct"
                ),
                "| method=",
                evaluation.get(
                    "method"
                ),
                "| conf=",
                evaluation.get(
                    "confidence"
                ),
            )


            if evaluation.get(
                "nli"
            ):

                print(
                    "  NLI:",
                    evaluation[
                        "nli"
                    ]
                )


    print(
        "\nSaved:",
        output_path
    )


if __name__ == "__main__":

    main()