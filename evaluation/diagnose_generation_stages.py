import json

from pathlib import Path
from types import SimpleNamespace

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)

from adaptive_agentic_rag.generation.citation import (
    validate_citations,
)


FROZEN_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "generation_stage_diagnostic_v2.json"
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
            "Matching frozen examples:",
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
# Context serialization
# ============================================================

def serialize_context(
    context,
):

    return {
        "total_words":
            context.total_words,

        "items": [

            {
                "citation_id":
                    item.citation_id,

                "chunk_id":
                    item.chunk_id,

                "document_id":
                    item.document_id,

                "title":
                    item.title,

                "source":
                    item.source,

                "url":
                    item.url,

                "score":
                    item.score,

                "text":
                    item.text,
            }

            for item
            in context.items
        ],
    }


# ============================================================
# Evidence path
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

            "context":
                serialize_context(
                    state[
                        "context"
                    ]
                ),
        }
    ]


    # ========================================================
    # One rewrite maximum
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

                "context":
                    serialize_context(
                        state[
                            "context"
                        ]
                    ),
            }
        )


    return (
        state,
        attempts,
    )


# ============================================================
# Yes / No evaluator for DIRECT_ANSWER
# ============================================================

def direct_answer_matches_gold(
    direct_answer,
    gold_answer,
):

    if not direct_answer:

        return None


    if not gold_answer:

        return None


    gold = (
        str(
            gold_answer
        )
        .strip()
        .lower()
    )


    direct = (
        str(
            direct_answer
        )
        .strip()
        .lower()
    )


    if gold == "yes":

        return direct.startswith(
            "yes"
        )


    if gold == "no":

        return direct.startswith(
            "no"
        )


    return None


# ============================================================
# Production-contract diagnostic
# ============================================================

def diagnose_generation(
    nodes,
    question,
    context,
):

    generator = (
        nodes.generator
    )


    # ========================================================
    # 1. Generate ONCE
    # ========================================================

    generator._load_generator()


    raw_answer = (
        generator._generate_draft(
            query=
                question,
            context=
                context,
            max_new_tokens=
                220,
        )
    )


    # ========================================================
    # 2. Parse DIRECT_ANSWER separately from FACTS
    #
    # THIS is the important difference from the old diagnostic.
    # ========================================================

    parsed = (
        generator._parse_draft(
            raw_answer
        )
    )


    # ========================================================
    # Model requested abstention
    # ========================================================

    model_insufficient = (
        parsed.direct_answer
        is not None
        and
        parsed.direct_answer
        .strip()
        .upper()
        ==
        "INSUFFICIENT_EVIDENCE"
    )


    if model_insufficient:

        return {
            "raw_answer":
                raw_answer,

            "direct_answer":
                parsed.direct_answer,

            "parsed_facts":
                parsed.evidence_claims,

            "model_insufficient":
                True,

            "claims":
                [],

            "final_answer":
                None,

            "citation_valid":
                None,

            "cited_ids":
                [],

            "runtime_grade":
                None,
        }


    # ========================================================
    # No facts
    # ========================================================

    if not parsed.evidence_claims:

        return {
            "raw_answer":
                raw_answer,

            "direct_answer":
                parsed.direct_answer,

            "parsed_facts":
                [],

            "model_insufficient":
                False,

            "claims":
                [],

            "final_answer":
                None,

            "citation_valid":
                False,

            "cited_ids":
                [],

            "runtime_grade":
                None,
        }


    # ========================================================
    # 3. Ground FACTS ONLY
    #
    # DIRECT_ANSWER never enters NLI.
    # ========================================================

    generator._load_claim_grounder()


    grounding_input = (
        generator
        ._render_claims_for_grounding(
            parsed.evidence_claims
        )
    )


    grounded = (
        generator
        .claim_grounder
        .ground(
            answer=
                grounding_input,
            context=
                context,
        )
    )


    # ========================================================
    # 4. Relevance over supported factual claims
    # ========================================================

    relevance = (
        generator
        .relevance_filter
        .filter(
            query=
                question,
            grounded_claims=
                grounded,
        )
    )


    relevant_map = {

        claim.claim:
            claim

        for claim
        in relevance.relevant_claims
    }


    filtered_map = {

        claim.claim:
            claim

        for claim
        in relevance.filtered_claims
    }


    claim_rows = []


    for claim in grounded.claims:

        relevance_score = None

        relevance_decision = (
            "NOT_RANKED"
        )


        if (
            claim.claim
            in relevant_map
        ):

            relevant_claim = (
                relevant_map[
                    claim.claim
                ]
            )


            relevance_score = (
                relevant_claim
                .relevance_score
            )


            relevance_decision = (
                "KEEP"
            )


        elif (
            claim.claim
            in filtered_map
        ):

            filtered_claim = (
                filtered_map[
                    claim.claim
                ]
            )


            relevance_score = (
                filtered_claim
                .relevance_score
            )


            relevance_decision = (
                "FILTER"
            )


        claim_rows.append(
            {
                "claim":
                    claim.claim,

                "supported":
                    claim.supported,

                "citation_id":
                    claim.citation_id,

                "label":
                    claim.label,

                "entailment_score":
                    claim.entailment_score,

                "supporting_text":
                    claim.supporting_text,

                "evidence_relevance_score":
                    getattr(
                        claim,
                        "evidence_relevance_score",
                        None,
                    ),

                "premise_mode":
                    getattr(
                        claim,
                        "premise_mode",
                        None,
                    ),

                "final_relevance_score":
                    relevance_score,

                "relevance_decision":
                    relevance_decision,
            }
        )


    # ========================================================
    # No grounded relevant facts
    # ========================================================

    if not relevance.relevant_claims:

        return {
            "raw_answer":
                raw_answer,

            "direct_answer":
                parsed.direct_answer,

            "parsed_facts":
                parsed.evidence_claims,

            "model_insufficient":
                False,

            "supported_count":
                grounded.supported_count,

            "unsupported_count":
                grounded.unsupported_count,

            "claims":
                claim_rows,

            "final_answer":
                None,

            "citation_valid":
                False,

            "cited_ids":
                [],

            "runtime_grade":
                None,
        }


    # ========================================================
    # 5. Build final answer using production contract
    # ========================================================

    final_answer = (
        generator
        ._build_grounded_answer(
            direct_answer=
                parsed.direct_answer,
            relevant_claims=
                relevance.relevant_claims,
        )
    )


    # ========================================================
    # 6. Citation validation
    # ========================================================

    citation_result = (
        validate_citations(
            answer=
                final_answer,
            context=
                context,
        )
    )


    # ========================================================
    # 7. Runtime AnswerGrader
    #
    # We avoid a second generation call.
    # ========================================================

    generation_result = (
        SimpleNamespace(
            answer=
                final_answer,

            raw_answer=
                raw_answer,

            direct_answer=
                parsed.direct_answer,

            abstained=
                False,

            citation_valid=
                citation_result.valid,

            cited_ids=
                citation_result.cited_ids,

            invalid_citation_ids=
                citation_result.invalid_ids,

            supported_claims=
                grounded.supported_count,

            unsupported_claims=
                grounded.unsupported_count,

            relevant_claims=
                len(
                    relevance.relevant_claims
                ),

            filtered_irrelevant_claims=
                len(
                    relevance.filtered_claims
                ),
        )
    )


    runtime_grade = (
        nodes.answer_grader.grade(
            query=
                question,
            generation_result=
                generation_result,
            evidence_sufficient=
                True,
        )
    )


    return {
        "raw_answer":
            raw_answer,

        "direct_answer":
            parsed.direct_answer,

        "parsed_facts":
            parsed.evidence_claims,

        "model_insufficient":
            False,

        "supported_count":
            grounded.supported_count,

        "unsupported_count":
            grounded.unsupported_count,

        "relevant_count":
            len(
                relevance.relevant_claims
            ),

        "filtered_count":
            len(
                relevance.filtered_claims
            ),

        "claims":
            claim_rows,

        "final_answer":
            final_answer,

        "citation_valid":
            citation_result.valid,

        "cited_ids":
            citation_result.cited_ids,

        "invalid_citation_ids":
            citation_result.invalid_ids,

        "runtime_grade":
            {
                "passed":
                    runtime_grade.passed,

                "supported_claim_ratio":
                    runtime_grade
                    .supported_claim_ratio,

                "relevance_score":
                    runtime_grade
                    .relevance_score,

                "reasons":
                    runtime_grade.reasons,
            },
    }


# ============================================================
# Console
# ============================================================

def print_diagnostic(
    diagnostic,
):

    print(
        "\n===== RAW GENERATION ====="
    )

    print(
        diagnostic[
            "raw_answer"
        ]
    )


    print(
        "\n===== PARSED CONTRACT ====="
    )


    print(
        "DIRECT ANSWER:",
        diagnostic[
            "direct_answer"
        ]
    )


    print(
        "FACT COUNT:",
        len(
            diagnostic[
                "parsed_facts"
            ]
        )
    )


    for index, fact in enumerate(
        diagnostic[
            "parsed_facts"
        ],
        start=1,
    ):

        print(
            f"FACT {index}:",
            fact,
        )


    print(
        "\n===== FACT GROUNDING ====="
    )


    for index, claim in enumerate(
        diagnostic[
            "claims"
        ],
        start=1,
    ):

        print(
            f"\nFACT {index}"
        )

        print(
            "Text:",
            claim[
                "claim"
            ]
        )

        print(
            "Supported:",
            claim[
                "supported"
            ]
        )

        print(
            "NLI:",
            claim[
                "label"
            ]
        )

        print(
            "Entailment:",
            claim[
                "entailment_score"
            ]
        )

        print(
            "Evidence relevance:",
            claim[
                "evidence_relevance_score"
            ]
        )

        print(
            "Premise mode:",
            claim[
                "premise_mode"
            ]
        )

        print(
            "Citation:",
            claim[
                "citation_id"
            ]
        )

        print(
            "Final relevance:",
            claim[
                "final_relevance_score"
            ]
        )

        print(
            "Decision:",
            claim[
                "relevance_decision"
            ]
        )


        if claim[
            "supporting_text"
        ]:

            print(
                "Best premise:"
            )

            print(
                claim[
                    "supporting_text"
                ]
            )


    print(
        "\n===== FINAL ANSWER ====="
    )


    print(
        diagnostic[
            "final_answer"
        ]
    )


    print(
        "Citation valid:",
        diagnostic[
            "citation_valid"
        ]
    )


    print(
        "Cited IDs:",
        diagnostic[
            "cited_ids"
        ]
    )


    print(
        "Runtime grade:",
        diagnostic[
            "runtime_grade"
        ]
    )


# ============================================================
# Main
# ============================================================

def main():

    examples = (
        load_examples()
    )


    selected = (
        find_target_examples(
            examples
        )
    )


    nodes = (
        RAGNodes()
    )


    output = []


    summary = {
        "cases":
            0,

        "evidence_sufficient":
            0,

        "structured_generation":
            0,

        "final_answers":
            0,

        "runtime_passes":
            0,

        "yes_no_direct_answer_scored":
            0,

        "yes_no_direct_answer_correct":
            0,

        "supported_facts":
            0,

        "unsupported_facts":
            0,
    }


    try:

        for (
            target_name,
            example,
        ) in selected:

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
                attempts,
            ) = prepare_evidence(
                nodes=
                    nodes,
                question=
                    question,
            )


            print(
                "\n===== EVIDENCE ====="
            )


            for attempt in attempts:

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


            record = {
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

                "question_type":
                    example.get(
                        "question_type"
                    ),

                "evidence_attempts":
                    attempts,

                "final_evidence_sufficient":
                    state[
                        "evidence_sufficient"
                    ],
            }


            if not state[
                "evidence_sufficient"
            ]:

                print(
                    "\nSKIP GENERATION:"
                )

                print(
                    "Evidence insufficient."
                )


                record[
                    "generation"
                ] = None


                output.append(
                    record
                )

                continue


            summary[
                "evidence_sufficient"
            ] += 1


            diagnostic = (
                diagnose_generation(
                    nodes=
                        nodes,
                    question=
                        question,
                    context=
                        state[
                            "context"
                        ],
                )
            )


            print_diagnostic(
                diagnostic
            )


            if (
                diagnostic[
                    "direct_answer"
                ]
                is not None
                and
                diagnostic[
                    "parsed_facts"
                ]
            ):

                summary[
                    "structured_generation"
                ] += 1


            summary[
                "supported_facts"
            ] += (
                diagnostic.get(
                    "supported_count",
                    0,
                )
            )


            summary[
                "unsupported_facts"
            ] += (
                diagnostic.get(
                    "unsupported_count",
                    0,
                )
            )


            if (
                diagnostic[
                    "final_answer"
                ]
                is not None
            ):

                summary[
                    "final_answers"
                ] += 1


            runtime_grade = (
                diagnostic[
                    "runtime_grade"
                ]
            )


            if (
                runtime_grade
                and
                runtime_grade[
                    "passed"
                ]
            ):

                summary[
                    "runtime_passes"
                ] += 1


            direct_correct = (
                direct_answer_matches_gold(
                    direct_answer=
                        diagnostic[
                            "direct_answer"
                        ],
                    gold_answer=
                        gold_answer,
                )
            )


            if (
                direct_correct
                is not None
            ):

                summary[
                    "yes_no_direct_answer_scored"
                ] += 1


                if direct_correct:

                    summary[
                        "yes_no_direct_answer_correct"
                    ] += 1


            record[
                "direct_answer_matches_gold"
            ] = direct_correct


            record[
                "generation"
            ] = diagnostic


            output.append(
                record
            )


    finally:

        nodes.close()


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


    scored = (
        summary[
            "yes_no_direct_answer_scored"
        ]
    )


    if scored:

        summary[
            "yes_no_direct_answer_accuracy"
        ] = (
            summary[
                "yes_no_direct_answer_correct"
            ]
            /
            scored
        )

    else:

        summary[
            "yes_no_direct_answer_accuracy"
        ] = None


    print(
        json.dumps(
            summary,
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
            {
                "summary":
                    summary,

                "records":
                    output,
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