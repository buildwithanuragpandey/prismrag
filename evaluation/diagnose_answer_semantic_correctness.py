import json
import re

from collections import Counter
from pathlib import Path


RESULTS_PATH = Path(
    "evaluation/results/"
    "e2e_smoke_20_results.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "answer_semantic_correctness_diagnostic.json"
)


# ============================================================
# Loading
# ============================================================

def load_records():

    with open(
        RESULTS_PATH,
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


    if isinstance(
        payload,
        dict,
    ):

        for key in (
            "records",
            "examples",
            "data",
        ):

            value = (
                payload.get(
                    key
                )
            )


            if isinstance(
                value,
                list,
            ):

                return value


    raise ValueError(
        (
            "Unsupported E2E results structure: "
            f"{type(payload).__name__}"
        )
    )


# ============================================================
# Normalization
# ============================================================

def normalize_text(
    value,
) -> str:

    if value is None:

        return ""


    text = (
        str(
            value
        )
        .lower()
        .strip()
    )


    text = re.sub(
        r"\[\d+\]",
        " ",
        text,
    )


    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )


    text = re.sub(
        r"\s+",
        " ",
        text,
    )


    return (
        text.strip()
    )


# ============================================================
# Yes / No label
# ============================================================

def yes_no_label(
    value,
) -> str | None:

    normalized = (
        normalize_text(
            value
        )
    )


    if not normalized:

        return None


    first_token = (
        normalized
        .split()[
            0
        ]
    )


    if first_token == "yes":

        return "yes"


    if first_token == "no":

        return "no"


    return None


# ============================================================
# Answer family
# ============================================================

def answer_family(
    gold_answer,
) -> str:

    if (
        yes_no_label(
            gold_answer
        )
        is not None
    ):

        return "yes_no"


    normalized = (
        normalize_text(
            gold_answer
        )
    )


    if not normalized:

        return "missing_gold"


    token_count = (
        len(
            normalized.split()
        )
    )


    if token_count <= 6:

        return "short_entity_or_value"


    return "free_form"


# ============================================================
# Resolver-effect classification
#
# NOTE:
#
# This remains an observable proxy:
#
# draft direct answer
#       ↓
# final direct answer
#
# If the two differ, some downstream generation logic changed
# the direct answer.
#
# In the current architecture this is primarily the
# RelationAwareAnswerResolver, but this diagnostic still
# treats it as telemetry rather than absolute proof.
# ============================================================

def resolver_effect(
    *,
    gold_answer,
    draft_direct_answer,
    final_direct_answer,
) -> str:

    gold_label = (
        yes_no_label(
            gold_answer
        )
    )


    draft_label = (
        yes_no_label(
            draft_direct_answer
        )
    )


    final_label = (
        yes_no_label(
            final_direct_answer
        )
    )


    if gold_label is None:

        return "not_yes_no_gold"


    if (
        draft_label is None
        and
        final_label is None
    ):

        return "no_yes_no_prediction"


    if (
        draft_label
        ==
        final_label
    ):

        if (
            final_label
            ==
            gold_label
        ):

            return "unchanged_correct"


        return "unchanged_wrong"


    if (
        draft_label
        !=
        gold_label
        and
        final_label
        ==
        gold_label
    ):

        return "corrected_wrong_to_right"


    if (
        draft_label
        ==
        gold_label
        and
        final_label
        !=
        gold_label
    ):

        return "harmed_right_to_wrong"


    if (
        final_label
        ==
        gold_label
    ):

        return "changed_to_correct"


    return "changed_but_still_wrong"


# ============================================================
# Runtime semantic category
# ============================================================

def semantic_runtime_category(
    record,
) -> str:

    is_null = (
        record.get(
            "question_type"
        )
        ==
        "null_query"
    )


    abstained = bool(
        record.get(
            "abstained",
            False,
        )
    )


    runtime_passed = bool(
        record.get(
            "runtime_grader_passed",
            False,
        )
    )


    smoke_correct = (
        record.get(
            "smoke_answer_correct"
        )
    )


    if is_null:

        if abstained:

            return "null_abstention"


        if runtime_passed:

            return "null_answer_runtime_pass"


        return "null_answer_runtime_fail"


    # ========================================================
    # Answerable
    # ========================================================

    if abstained:

        return "answerable_abstention"


    if smoke_correct is True:

        if runtime_passed:

            return "correct_answer_runtime_pass"


        return "correct_answer_runtime_fail"


    if smoke_correct is False:

        if runtime_passed:

            return "wrong_answer_runtime_pass"


        return "wrong_answer_runtime_fail"


    if runtime_passed:

        return "unscored_answer_runtime_pass"


    return "unscored_answer_runtime_fail"


# ============================================================
# Analyze one record
# ============================================================

def analyze_record(
    record,
) -> dict:

    gold_answer = (
        record.get(
            "gold_answer"
        )
    )


    draft_direct_answer = (
        record.get(
            "draft_direct_answer"
        )
    )


    final_direct_answer = (
        record.get(
            "direct_answer"
        )
    )


    family = (
        answer_family(
            gold_answer
        )
    )


    effect = (
        resolver_effect(
            gold_answer=
                gold_answer,

            draft_direct_answer=
                draft_direct_answer,

            final_direct_answer=
                final_direct_answer,
        )
    )


    runtime_category = (
        semantic_runtime_category(
            record
        )
    )


    gold_label = (
        yes_no_label(
            gold_answer
        )
    )


    draft_label = (
        yes_no_label(
            draft_direct_answer
        )
    )


    final_label = (
        yes_no_label(
            final_direct_answer
        )
    )


    polarity_correct = None


    if (
        gold_label is not None
        and
        final_label is not None
    ):

        polarity_correct = (
            gold_label
            ==
            final_label
        )


    return {
        "id":
            record.get(
                "id"
            ),

        "question_type":
            record.get(
                "question_type"
            ),

        "question":
            record.get(
                "question"
            ),

        "gold_answer":
            gold_answer,

        "answer_family":
            family,

        "draft_direct_answer":
            draft_direct_answer,

        "final_direct_answer":
            final_direct_answer,

        "gold_yes_no_label":
            gold_label,

        "draft_yes_no_label":
            draft_label,

        "final_yes_no_label":
            final_label,

        "resolver_effect":
            effect,

        "polarity_correct":
            polarity_correct,

        "abstained":
            bool(
                record.get(
                    "abstained",
                    False,
                )
            ),

        "runtime_grader_passed":
            bool(
                record.get(
                    "runtime_grader_passed",
                    False,
                )
            ),

        "runtime_relevance_score":
            record.get(
                "runtime_relevance_score"
            ),

        "smoke_answer_correct":
            record.get(
                "smoke_answer_correct"
            ),

        "citation_valid":
            bool(
                record.get(
                    "citation_valid",
                    False,
                )
            ),

        "supported_claims":
            record.get(
                "supported_claims"
            ),

        "relevant_claims":
            record.get(
                "relevant_claims"
            ),

        "semantic_runtime_category":
            runtime_category,
    }


# ============================================================
# Summary
# ============================================================

def summarize(
    records,
):

    analyzed = [
        analyze_record(
            record
        )

        for record
        in records
    ]


    answerable = [
        record

        for record
        in analyzed

        if (
            record[
                "question_type"
            ]
            !=
            "null_query"
        )
    ]


    answered_answerable = [
        record

        for record
        in answerable

        if not record[
            "abstained"
        ]
    ]


    runtime_passed_answered = [
        record

        for record
        in answered_answerable

        if record[
            "runtime_grader_passed"
        ]
    ]


    wrong_runtime_passes = [
        record

        for record
        in runtime_passed_answered

        if (
            record[
                "smoke_answer_correct"
            ]
            is False
        )
    ]


    yes_no_answerable = [
        record

        for record
        in answerable

        if (
            record[
                "answer_family"
            ]
            ==
            "yes_no"
        )
    ]


    answered_yes_no = [
        record

        for record
        in yes_no_answerable

        if not record[
            "abstained"
        ]
    ]


    yes_no_with_prediction = [
        record

        for record
        in answered_yes_no

        if (
            record[
                "final_yes_no_label"
            ]
            is not None
        )
    ]


    correct_yes_no = [
        record

        for record
        in yes_no_with_prediction

        if (
            record[
                "polarity_correct"
            ]
            is True
        )
    ]


    wrong_yes_no = [
        record

        for record
        in yes_no_with_prediction

        if (
            record[
                "polarity_correct"
            ]
            is False
        )
    ]


    resolver_corrected = [
        record

        for record
        in analyzed

        if (
            record[
                "resolver_effect"
            ]
            ==
            "corrected_wrong_to_right"
        )
    ]


    resolver_harmed = [
        record

        for record
        in analyzed

        if (
            record[
                "resolver_effect"
            ]
            ==
            "harmed_right_to_wrong"
        )
    ]


    unresolved_wrong_polarity_runtime_pass = [
        record

        for record
        in wrong_yes_no

        if record[
            "runtime_grader_passed"
        ]
    ]


    wrong_runtime_pass_by_family = (
        Counter(
            record[
                "answer_family"
            ]

            for record
            in wrong_runtime_passes
        )
    )


    runtime_category_distribution = (
        Counter(
            record[
                "semantic_runtime_category"
            ]

            for record
            in analyzed
        )
    )


    resolver_effect_distribution = (
        Counter(
            record[
                "resolver_effect"
            ]

            for record
            in analyzed
        )
    )


    by_question_type = {}


    for question_type in sorted(
        set(
            record[
                "question_type"
            ]

            for record
            in analyzed
        )
    ):

        subset = [
            record

            for record
            in analyzed

            if (
                record[
                    "question_type"
                ]
                ==
                question_type
            )
        ]


        by_question_type[
            question_type
        ] = {
            "count":
                len(
                    subset
                ),

            "answered":
                sum(
                    not record[
                        "abstained"
                    ]

                    for record
                    in subset
                ),

            "runtime_passed":
                sum(
                    record[
                        "runtime_grader_passed"
                    ]

                    for record
                    in subset
                ),

            "smoke_correct":
                sum(
                    record[
                        "smoke_answer_correct"
                    ]
                    is True

                    for record
                    in subset
                ),

            "wrong_runtime_pass":
                sum(
                    (
                        not record[
                            "abstained"
                        ]
                        and
                        record[
                            "runtime_grader_passed"
                        ]
                        and
                        record[
                            "smoke_answer_correct"
                        ]
                        is False
                    )

                    for record
                    in subset
                ),
        }


    summary = {
        "total_records":
            len(
                analyzed
            ),

        "answerable_records":
            len(
                answerable
            ),

        "answered_answerable":
            len(
                answered_answerable
            ),

        # ----------------------------------------------------
        # Runtime grader semantic blind spot
        # ----------------------------------------------------

        "answered_runtime_passes":
            len(
                runtime_passed_answered
            ),

        "wrong_answer_runtime_passes":
            len(
                wrong_runtime_passes
            ),

        "runtime_pass_semantic_false_positive_proxy_rate":
            (
                len(
                    wrong_runtime_passes
                )
                /
                len(
                    runtime_passed_answered
                )

                if runtime_passed_answered
                else None
            ),

        "wrong_runtime_pass_by_answer_family":
            dict(
                wrong_runtime_pass_by_family
            ),

        # ----------------------------------------------------
        # Yes/no polarity
        # ----------------------------------------------------

        "yes_no_answerable":
            len(
                yes_no_answerable
            ),

        "answered_yes_no":
            len(
                answered_yes_no
            ),

        "yes_no_with_prediction":
            len(
                yes_no_with_prediction
            ),

        "yes_no_polarity_correct":
            len(
                correct_yes_no
            ),

        "yes_no_polarity_wrong":
            len(
                wrong_yes_no
            ),

        "yes_no_polarity_accuracy":
            (
                len(
                    correct_yes_no
                )
                /
                len(
                    yes_no_with_prediction
                )

                if yes_no_with_prediction
                else None
            ),

        # ----------------------------------------------------
        # Relation resolver observable effect
        # ----------------------------------------------------

        "resolver_corrected_wrong_to_right":
            len(
                resolver_corrected
            ),

        "resolver_harmed_right_to_wrong":
            len(
                resolver_harmed
            ),

        "resolver_effect_distribution":
            dict(
                resolver_effect_distribution
            ),

        # ----------------------------------------------------
        # Remaining dangerous family
        # ----------------------------------------------------

        "unresolved_wrong_polarity_runtime_passes":
            len(
                unresolved_wrong_polarity_runtime_pass
            ),

        "unresolved_wrong_polarity_runtime_pass_ids": [
            record[
                "id"
            ]

            for record
            in unresolved_wrong_polarity_runtime_pass
        ],

        # ----------------------------------------------------
        # General
        # ----------------------------------------------------

        "runtime_category_distribution":
            dict(
                runtime_category_distribution
            ),

        "by_question_type":
            by_question_type,
    }


    return (
        analyzed,
        summary,
    )


# ============================================================
# Console failure report
# ============================================================

def print_failures(
    analyzed,
):

    dangerous = [
        record

        for record
        in analyzed

        if (
            not record[
                "abstained"
            ]
            and
            record[
                "runtime_grader_passed"
            ]
            and
            record[
                "smoke_answer_correct"
            ]
            is False
        )
    ]


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "ANSWERED + RUNTIME PASS + GOLD-SIDE WRONG"
    )


    print(
        "=" * 100
    )


    if not dangerous:

        print(
            "None."
        )

        return


    for (
        index,
        record,
    ) in enumerate(
        dangerous,
        start=1,
    ):

        print(
            "\n"
            +
            "-" * 100
        )


        print(
            (
                f"{index}/"
                f"{len(dangerous)} "
                f"| {record['id']} "
                f"| {record['question_type']} "
                f"| {record['answer_family']}"
            )
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
            "\nGold:"
        )


        print(
            record[
                "gold_answer"
            ]
        )


        print(
            "\nDraft direct answer:"
        )


        print(
            record[
                "draft_direct_answer"
            ]
        )


        print(
            "\nFinal direct answer:"
        )


        print(
            record[
                "final_direct_answer"
            ]
        )


        print(
            "\nResolver effect:",
            record[
                "resolver_effect"
            ],
        )


        print(
            "Runtime relevance:",
            record[
                "runtime_relevance_score"
            ],
        )


        print(
            "Citation valid:",
            record[
                "citation_valid"
            ],
        )


        print(
            "Supported claims:",
            record[
                "supported_claims"
            ],
        )


        print(
            "Relevant claims:",
            record[
                "relevant_claims"
            ],
        )


# ============================================================
# Main
# ============================================================

def main():

    records = (
        load_records()
    )


    (
        analyzed,
        summary,
    ) = (
        summarize(
            records
        )
    )


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "ANSWER SEMANTIC CORRECTNESS DIAGNOSTIC"
    )


    print(
        "=" * 100
    )


    print(
        "Input:",
        RESULTS_PATH
    )


    print(
        "Model execution:",
        "NONE"
    )


    print(
        "Production files modified:",
        "NO"
    )


    print_failures(
        analyzed
    )


    print(
        "\n"
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
                "input":
                    str(
                        RESULTS_PATH
                    ),

                "warning": (
                    "Smoke correctness is a lightweight "
                    "gold-side diagnostic, not the final "
                    "answer-quality benchmark."
                ),

                "summary":
                    summary,

                "records":
                    analyzed,
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