import json
import time
import argparse
from pathlib import Path

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)


FROZEN_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)


def output_path_for_model(
    model_name: str,
) -> Path:

    slug = (
        model_name
        .replace("/", "__")
        .replace("\\", "__")
    )

    return Path(
        "evaluation/results"
    ) / (
        f"self_correction_{slug}.json"
    )


TARGETS = {
    "taylor_kelce": [
        "taylor",
        "kelce",
    ],

    "sam_altman": [
        "sam altman",
    ],

    "mctominay_haaland": [
        "mctominay",
        "haaland",
    ],

    "google_youtube": [
        "google",
        "youtube",
    ],

    "google_epic": [
        "google",
        "epic",
    ],
}


# ============================================================
# Dataset
# ============================================================

def load_examples():

    with open(
        FROZEN_PATH,
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
        "Could not locate frozen examples."
    )


def find_target_examples(
    examples,
):

    selected = []


    for (
        target_name,
        required_terms,
    ) in TARGETS.items():

        matches = []


        for example in examples:

            question = (
                example.get(
                    "question"
                )
                or
                example.get(
                    "query"
                )
                or
                ""
            )


            normalized = (
                question.lower()
            )


            if all(
                term in normalized
                for term in required_terms
            ):

                matches.append(
                    example
                )


        if not matches:

            print(
                f"[NOT FOUND] "
                f"{target_name}"
            )

            continue


        matches.sort(
            key=lambda item:
                len(
                    item.get(
                        "question",
                        ""
                    )
                )
        )


        chosen = (
            matches[
                0
            ]
        )


        print(
            f"\n[{target_name}]"
        )

        print(
            chosen[
                "question"
            ]
        )

        print(
            "Matches:",
            len(
                matches
            )
        )


        selected.append(
            (
                target_name,
                chosen,
            )
        )


    return selected


# ============================================================
# Evidence preparation
# ============================================================

def prepare_evidence(
    nodes,
    question,
):

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


    state.update(
        nodes.grade_evidence(
            state
        )
    )


    attempts = [
        {
            "attempt":
                0,

            "query":
                state[
                    "current_query"
                ],

            "evidence_sufficient":
                state[
                    "evidence_sufficient"
                ],

            "evidence_score":
                state[
                    "evidence_score"
                ],

            "evidence_reasons":
                state[
                    "evidence_reasons"
                ],
        }
    ]


    # ========================================================
    # Match current production architecture:
    # one rewrite maximum.
    # ========================================================

    if not state[
        "evidence_sufficient"
    ]:

        state.update(
            nodes.rewrite_query(
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


        state.update(
            nodes.grade_evidence(
                state
            )
        )


        attempts.append(
            {
                "attempt":
                    state[
                        "retry_count"
                    ],

                "query":
                    state[
                        "current_query"
                    ],

                "evidence_sufficient":
                    state[
                        "evidence_sufficient"
                    ],

                "evidence_score":
                    state[
                        "evidence_score"
                    ],

                "evidence_reasons":
                    state[
                        "evidence_reasons"
                    ],
            }
        )


    return (
        state,
        attempts,
    )


# ============================================================
# Yes / No helpers
# ============================================================

def yes_no_label(
    answer,
):

    if not answer:

        return None


    normalized = (
        str(
            answer
        )
        .strip()
        .lower()
    )


    if normalized.startswith(
        "yes"
    ):

        return "yes"


    if normalized.startswith(
        "no"
    ):

        return "no"


    return None


def answer_correct(
    answer,
    gold_answer,
):

    prediction = (
        yes_no_label(
            answer
        )
    )


    gold = (
        yes_no_label(
            gold_answer
        )
    )


    if (
        prediction is None
        or
        gold is None
    ):

        return None


    return (
        prediction
        ==
        gold
    )


def correction_outcome(
    draft_correct,
    final_correct,
):

    if (
        draft_correct is None
        or
        final_correct is None
    ):

        return (
            "unscored"
        )


    if (
        draft_correct is False
        and
        final_correct is True
    ):

        return (
            "corrected_error"
        )


    if (
        draft_correct is True
        and
        final_correct is False
    ):

        return (
            "introduced_error"
        )


    if (
        draft_correct is True
        and
        final_correct is True
    ):

        return (
            "preserved_correct"
        )


    return (
        "preserved_wrong"
    )
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )

    return parser.parse_args()

# ============================================================
# Main
# ============================================================

def main():

    examples = (
        load_examples()
    )
    args = parse_args()
    output_path = (
        output_path_for_model(
            args.model
        )
    )
    print(
        "\nGenerator model:",
        args.model,
    )

    examples = load_examples()

    targets = (
        find_target_examples(
            examples
        )
    )


    nodes = (
        RAGNodes()
    )

    nodes.generator.model_name = (
        args.model
    )
    records = []


    summary = {
        "cases":
            0,

        "evidence_sufficient":
            0,

        "answers_generated":
            0,

        "abstentions":
            0,

        "draft_scored":
            0,

        "draft_correct":
            0,

        "final_scored":
            0,

        "final_correct":
            0,

        "corrected_errors":
            0,

        "introduced_errors":
            0,

        "preserved_correct":
            0,

        "preserved_wrong":
            0,

        "runtime_passes":
            0,

        "supported_claims":
            0,

        "unsupported_claims":
            0,

        "relevant_claims":
            0,

        "generation_seconds":
            [],
    }


    try:

        for (
            target_name,
            example,
        ) in targets:

            summary[
                "cases"
            ] += 1


            question = (
                example[
                    "question"
                ]
            )


            gold_answer = (
                example.get(
                    "answer"
                )
            )


            print(
                "\n"
                +
                "=" * 100
            )

            print(
                target_name.upper()
            )

            print(
                "=" * 100
            )


            print(
                "QUESTION:"
            )

            print(
                question
            )


            print(
                "\nGOLD:"
            )

            print(
                gold_answer
            )


            (
                state,
                evidence_attempts,
            ) = prepare_evidence(
                nodes=
                    nodes,
                question=
                    question,
            )


            print(
                "\n===== EVIDENCE ====="
            )


            for attempt in (
                evidence_attempts
            ):

                print(
                    (
                        f"Attempt "
                        f"{attempt['attempt']} | "
                        f"score="
                        f"{attempt['evidence_score']:.4f} | "
                        f"sufficient="
                        f"{attempt['evidence_sufficient']}"
                    )
                )


            evidence_sufficient = (
                state[
                    "evidence_sufficient"
                ]
            )


            if evidence_sufficient:

                summary[
                    "evidence_sufficient"
                ] += 1


            # =================================================
            # IMPORTANT:
            #
            # Use the PUBLIC production generation path.
            #
            # This includes:
            #
            # Pass 1
            # citation-linked facts
            # grounding
            # relevance
            # Pass 2 self-correction
            # final citations
            # =================================================

            start_time = (
                time.perf_counter()
            )


            generation_result = (
                nodes.generator.generate(
                    query=
                        question,
                    context=
                        state[
                            "context"
                        ],
                    evidence_sufficient=
                        evidence_sufficient,
                )
            )


            generation_seconds = (
                time.perf_counter()
                -
                start_time
            )


            summary[
                "generation_seconds"
            ].append(
                generation_seconds
            )


            # =================================================
            # Parse Pass-1 output only for diagnostics.
            # =================================================

            parsed = None


            if (
                generation_result
                .raw_answer
            ):

                parsed = (
                    nodes.generator
                    ._parse_draft(
                        generation_result
                        .raw_answer
                    )
                )


            draft_answer = (
                generation_result
                .draft_direct_answer
            )


            final_direct_answer = (
                generation_result
                .direct_answer
            )


            draft_correct = (
                answer_correct(
                    answer=
                        draft_answer,
                    gold_answer=
                        gold_answer,
                )
            )


            final_correct = (
                answer_correct(
                    answer=
                        final_direct_answer,
                    gold_answer=
                        gold_answer,
                )
            )


            outcome = (
                correction_outcome(
                    draft_correct=
                        draft_correct,
                    final_correct=
                        final_correct,
                )
            )


            # =================================================
            # Runtime AnswerGrader
            # =================================================

            runtime_grade = (
                nodes.answer_grader.grade(
                    query=
                        question,
                    generation_result=
                        generation_result,
                    evidence_sufficient=
                        evidence_sufficient,
                )
            )


            # =================================================
            # Summary counters
            # =================================================

            if (
                generation_result
                .abstained
            ):

                summary[
                    "abstentions"
                ] += 1

            else:

                summary[
                    "answers_generated"
                ] += 1


            if (
                draft_correct
                is not None
            ):

                summary[
                    "draft_scored"
                ] += 1


                if draft_correct:

                    summary[
                        "draft_correct"
                    ] += 1


            if (
                final_correct
                is not None
            ):

                summary[
                    "final_scored"
                ] += 1


                if final_correct:

                    summary[
                        "final_correct"
                    ] += 1


            if outcome in summary:

                summary[
                    outcome
                ] += 1


            if (
                runtime_grade
                .passed
            ):

                summary[
                    "runtime_passes"
                ] += 1


            summary[
                "supported_claims"
            ] += int(
                generation_result
                .supported_claims
            )


            summary[
                "unsupported_claims"
            ] += int(
                generation_result
                .unsupported_claims
            )


            summary[
                "relevant_claims"
            ] += int(
                generation_result
                .relevant_claims
            )


            # =================================================
            # Console
            # =================================================

            print(
                "\n===== PASS 1 ====="
            )


            print(
                "Draft answer:",
                draft_answer
            )


            if parsed:

                for (
                    index,
                    fact,
                ) in enumerate(
                    parsed.evidence_facts,
                    start=1,
                ):

                    print(
                        (
                            f"FACT {index} | "
                            f"citation="
                            f"{fact.citation_id}"
                        )
                    )

                    print(
                        fact.text
                    )


            print(
                "\n===== PASS 2 ====="
            )


            print(
                "Final direct answer:",
                final_direct_answer
            )


            print(
                "\n===== FINAL ANSWER ====="
            )

            print(
                generation_result
                .answer
            )


            print(
                "\n===== QUALITY ====="
            )


            print(
                "Draft correct:",
                draft_correct
            )

            print(
                "Final correct:",
                final_correct
            )

            print(
                "Self-correction outcome:",
                outcome
            )

            print(
                "Abstained:",
                generation_result
                .abstained
            )

            print(
                "Supported claims:",
                generation_result
                .supported_claims
            )

            print(
                "Unsupported claims:",
                generation_result
                .unsupported_claims
            )

            print(
                "Relevant claims:",
                generation_result
                .relevant_claims
            )

            print(
                "Citation valid:",
                generation_result
                .citation_valid
            )

            print(
                "Runtime passed:",
                runtime_grade
                .passed
            )

            print(
                "Runtime relevance:",
                runtime_grade
                .relevance_score
            )

            print(
                "Generation seconds:",
                round(
                    generation_seconds,
                    3,
                )
            )


            records.append(
                {
                    "target":
                        target_name,

                    "example_id":
                        example.get(
                            "id"
                        ),

                    "question":
                        question,

                    "gold_answer":
                        gold_answer,

                    "evidence_attempts":
                        evidence_attempts,

                    "evidence_sufficient":
                        evidence_sufficient,

                    "raw_answer":
                        generation_result
                        .raw_answer,

                    "draft_direct_answer":
                        draft_answer,

                    "final_direct_answer":
                        final_direct_answer,

                    "draft_correct":
                        draft_correct,

                    "final_correct":
                        final_correct,

                    "self_correction_outcome":
                        outcome,

                    "final_answer":
                        generation_result
                        .answer,

                    "abstained":
                        generation_result
                        .abstained,

                    "citation_valid":
                        generation_result
                        .citation_valid,

                    "cited_ids":
                        generation_result
                        .cited_ids,

                    "draft_claims":
                        generation_result
                        .draft_claims,

                    "supported_claims":
                        generation_result
                        .supported_claims,

                    "unsupported_claims":
                        generation_result
                        .unsupported_claims,

                    "relevant_claims":
                        generation_result
                        .relevant_claims,

                    "filtered_irrelevant_claims":
                        generation_result
                        .filtered_irrelevant_claims,

                    "runtime_grade": {
                        "passed":
                            runtime_grade
                            .passed,

                        "supported_claim_ratio":
                            runtime_grade
                            .supported_claim_ratio,

                        "relevance_score":
                            runtime_grade
                            .relevance_score,

                        "reasons":
                            runtime_grade
                            .reasons,
                    },

                    "generation_seconds":
                        round(
                            generation_seconds,
                            4,
                        ),
                }
            )


    finally:

        nodes.close()


    # ========================================================
    # Aggregate metrics
    # ========================================================

    if (
        summary[
            "draft_scored"
        ]
        >
        0
    ):

        summary[
            "draft_accuracy"
        ] = (
            summary[
                "draft_correct"
            ]
            /
            summary[
                "draft_scored"
            ]
        )

    else:

        summary[
            "draft_accuracy"
        ] = None


    if (
        summary[
            "final_scored"
        ]
        >
        0
    ):

        summary[
            "final_accuracy"
        ] = (
            summary[
                "final_correct"
            ]
            /
            summary[
                "final_scored"
            ]
        )

    else:

        summary[
            "final_accuracy"
        ] = None


    if (
        summary[
            "draft_accuracy"
        ]
        is not None
        and
        summary[
            "final_accuracy"
        ]
        is not None
    ):

        summary[
            "self_correction_gain"
        ] = (
            summary[
                "final_accuracy"
            ]
            -
            summary[
                "draft_accuracy"
            ]
        )

    else:

        summary[
            "self_correction_gain"
        ] = None


    generation_times = (
        summary.pop(
            "generation_seconds"
        )
    )


    if generation_times:

        summary[
            "mean_generation_seconds"
        ] = (
            sum(
                generation_times
            )
            /
            len(
                generation_times
            )
        )

        summary[
            "max_generation_seconds"
        ] = max(
            generation_times
        )

    else:

        summary[
            "mean_generation_seconds"
        ] = None

        summary[
            "max_generation_seconds"
        ] = None


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
        )
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "summary":
                    summary,

                "records":
                    records,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


    print(
        "\nSaved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":

    main()