import json
from pathlib import Path

import numpy as np

from sentence_transformers import CrossEncoder

from adaptive_agentic_rag.generation.claim_grounder import (
    NLI_MODEL_NAME,
)

from adaptive_agentic_rag.generation.sentence_splitter import (
    split_sentences,
)


INPUT_PATH = Path(
    "evaluation/results/"
    "generation_stage_diagnostic.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "nli_grounding_modes_diagnostic.json"
)


DEFAULT_LABELS = [
    "contradiction",
    "entailment",
    "neutral",
]


TOP_SINGLE_CANDIDATES = 8

MAX_PAIR_CANDIDATES = 24


# ============================================================
# Label helpers
# ============================================================

def get_labels(
    model,
) -> list[str]:

    config = getattr(
        getattr(
            model,
            "model",
            None,
        ),
        "config",
        None,
    )


    if config is None:

        return DEFAULT_LABELS


    id2label = getattr(
        config,
        "id2label",
        None,
    )


    if not id2label:

        return DEFAULT_LABELS


    labels = []


    for index in range(
        3
    ):

        label = (
            id2label.get(
                index
            )
            or
            id2label.get(
                str(
                    index
                )
            )
        )


        if label is None:

            return DEFAULT_LABELS


        labels.append(
            str(
                label
            ).lower()
        )


    return labels


# ============================================================
# Evidence-unit construction
# ============================================================

def build_evidence_units(
    context_items,
):

    units = []


    for item in context_items:

        citation_id = (
            item[
                "citation_id"
            ]
        )


        title = (
            item.get(
                "title",
                ""
            )
            or ""
        ).strip()


        source = (
            item.get(
                "source",
                ""
            )
            or ""
        ).strip()


        text = (
            item.get(
                "text",
                ""
            )
            or ""
        ).strip()


        sentences = (
            split_sentences(
                text,
                split_newlines=False,
                min_chars=20,
            )
        )


        provenance_parts = []


        if source:

            provenance_parts.append(
                f"Source: {source}."
            )


        if title:

            provenance_parts.append(
                f"Title: {title}."
            )


        provenance = " ".join(
            provenance_parts
        )


        for index, sentence in enumerate(
            sentences
        ):

            # =================================================
            # Single sentence
            # =================================================

            units.append(
                {
                    "citation_id":
                        citation_id,

                    "title":
                        title,

                    "source":
                        source,

                    "kind":
                        "sentence",

                    "text":
                        sentence,

                    "provenance_text":
                        (
                            f"{provenance} "
                            f"Evidence: {sentence}"
                        ).strip(),
                }
            )


            # =================================================
            # Two-sentence window
            # =================================================

            if (
                index + 1
                <
                len(
                    sentences
                )
            ):

                window = (
                    sentence
                    +
                    " "
                    +
                    sentences[
                        index + 1
                    ]
                )


                units.append(
                    {
                        "citation_id":
                            citation_id,

                        "title":
                            title,

                        "source":
                            source,

                        "kind":
                            "window",

                        "text":
                            window,

                        "provenance_text":
                            (
                                f"{provenance} "
                                f"Evidence: {window}"
                            ).strip(),
                    }
                )


    return units


# ============================================================
# Score arbitrary premises against one claim
# ============================================================

def score_premises(
    model,
    labels,
    claim,
    candidates,
    premise_key,
):

    if not candidates:

        return []


    pairs = [

        (
            candidate[
                premise_key
            ],
            claim,
        )

        for candidate
        in candidates
    ]


    probabilities = (
        model.predict(
            pairs,
            apply_softmax=True,
            batch_size=32,
            show_progress_bar=False,
        )
    )


    probabilities = np.asarray(
        probabilities
    )


    entailment_index = (
        labels.index(
            "entailment"
        )
    )


    rows = []


    for (
        candidate,
        probability,
    ) in zip(
        candidates,
        probabilities,
    ):

        predicted_index = int(
            np.argmax(
                probability
            )
        )


        rows.append(
            {
                **candidate,

                "label":
                    labels[
                        predicted_index
                    ],

                "contradiction_score":
                    round(
                        float(
                            probability[
                                labels.index(
                                    "contradiction"
                                )
                            ]
                        ),
                        4,
                    ),

                "entailment_score":
                    round(
                        float(
                            probability[
                                entailment_index
                            ]
                        ),
                        4,
                    ),

                "neutral_score":
                    round(
                        float(
                            probability[
                                labels.index(
                                    "neutral"
                                )
                            ]
                        ),
                        4,
                    ),
            }
        )


    rows.sort(
        key=lambda row:
            row[
                "entailment_score"
            ],
        reverse=True,
    )


    return rows


# ============================================================
# Build cross-citation evidence pairs
# ============================================================

def build_pair_candidates(
    ranked_single_units,
):

    top_units = (
        ranked_single_units[
            :TOP_SINGLE_CANDIDATES
        ]
    )


    pairs = []


    seen = set()


    for left_index in range(
        len(
            top_units
        )
    ):

        for right_index in range(
            left_index + 1,
            len(
                top_units
            )
        ):

            left = (
                top_units[
                    left_index
                ]
            )

            right = (
                top_units[
                    right_index
                ]
            )


            # =================================================
            # Multi-hop diagnostic:
            #
            # We specifically want evidence from DIFFERENT
            # citations.
            # =================================================

            if (
                left[
                    "citation_id"
                ]
                ==
                right[
                    "citation_id"
                ]
            ):

                continue


            citation_ids = tuple(
                sorted(
                    [
                        left[
                            "citation_id"
                        ],
                        right[
                            "citation_id"
                        ],
                    ]
                )
            )


            key = (
                citation_ids,
                left[
                    "provenance_text"
                ],
                right[
                    "provenance_text"
                ],
            )


            if key in seen:

                continue


            seen.add(
                key
            )


            premise = (
                "Evidence source A: "
                +
                left[
                    "provenance_text"
                ]
                +
                "\n"
                +
                "Evidence source B: "
                +
                right[
                    "provenance_text"
                ]
            )


            pairs.append(
                {
                    "citation_ids":
                        list(
                            citation_ids
                        ),

                    "kind":
                        "cross_citation_pair",

                    "premise":
                        premise,
                }
            )


            if (
                len(
                    pairs
                )
                >=
                MAX_PAIR_CANDIDATES
            ):

                return pairs


    return pairs


# ============================================================
# NLI model sanity test
# ============================================================

def run_sanity_check(
    model,
    labels,
):

    examples = [
        (
            "A man is eating pizza.",
            "A man is eating food.",
            "entailment",
        ),
        (
            "A man is eating pizza.",
            "A man is sleeping.",
            "contradiction",
        ),
        (
            "A man is eating pizza.",
            "The man likes football.",
            "neutral",
        ),
    ]


    pairs = [
        (
            premise,
            hypothesis,
        )

        for (
            premise,
            hypothesis,
            _
        )
        in examples
    ]


    scores = (
        model.predict(
            pairs,
            apply_softmax=True,
            show_progress_bar=False,
        )
    )


    scores = np.asarray(
        scores
    )


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "NLI SANITY CHECK"
    )

    print(
        "=" * 100
    )


    for (
        example,
        probability,
    ) in zip(
        examples,
        scores,
    ):

        premise, hypothesis, expected = (
            example
        )


        predicted = (
            labels[
                int(
                    np.argmax(
                        probability
                    )
                )
            ]
        )


        print(
            "\nPremise:",
            premise
        )

        print(
            "Hypothesis:",
            hypothesis
        )

        print(
            "Expected:",
            expected
        )

        print(
            "Predicted:",
            predicted
        )

        print(
            "Probabilities:",
            {
                labels[index]:
                    round(
                        float(
                            probability[
                                index
                            ]
                        ),
                        4,
                    )

                for index
                in range(
                    len(
                        labels
                    )
                )
            }
        )


# ============================================================
# Format best result
# ============================================================

def best_or_none(
    rows,
):

    if not rows:

        return None


    return rows[
        0
    ]


# ============================================================
# Diagnostic for one claim
# ============================================================

def diagnose_claim(
    model,
    labels,
    claim,
    context_items,
):

    units = (
        build_evidence_units(
            context_items
        )
    )


    # ========================================================
    # Mode A:
    # production-like plain sentence/window
    # ========================================================

    plain_results = (
        score_premises(
            model=
                model,
            labels=
                labels,
            claim=
                claim,
            candidates=
                units,
            premise_key=
                "text",
        )
    )


    # ========================================================
    # Mode B:
    # source + title + evidence
    # ========================================================

    provenance_results = (
        score_premises(
            model=
                model,
            labels=
                labels,
            claim=
                claim,
            candidates=
                units,
            premise_key=
                "provenance_text",
        )
    )


    # ========================================================
    # Mode C:
    # two high-ranking pieces from different citations
    # ========================================================

    pair_candidates = (
        build_pair_candidates(
            provenance_results
        )
    )


    pair_results = []


    if pair_candidates:

        pair_results = (
            score_premises(
                model=
                    model,
                labels=
                    labels,
                claim=
                    claim,
                candidates=
                    pair_candidates,
                premise_key=
                    "premise",
            )
        )


    return {
        "plain":
            best_or_none(
                plain_results
            ),

        "provenance":
            best_or_none(
                provenance_results
            ),

        "cross_citation_pair":
            best_or_none(
                pair_results
            ),
    }


# ============================================================
# Compact console printing
# ============================================================

def print_mode(
    name,
    result,
):

    print(
        f"\n{name}"
    )


    if result is None:

        print(
            "  No candidate."
        )

        return


    print(
        "  label:",
        result[
            "label"
        ]
    )

    print(
        "  contradiction:",
        result[
            "contradiction_score"
        ]
    )

    print(
        "  entailment:",
        result[
            "entailment_score"
        ]
    )

    print(
        "  neutral:",
        result[
            "neutral_score"
        ]
    )


    if (
        "citation_ids"
        in result
    ):

        print(
            "  citations:",
            result[
                "citation_ids"
            ]
        )

        print(
            "  premise:"
        )

        print(
            " ",
            result[
                "premise"
            ]
        )

    else:

        print(
            "  citation:",
            result[
                "citation_id"
            ]
        )

        print(
            "  source:",
            result[
                "source"
            ]
        )

        print(
            "  title:",
            result[
                "title"
            ]
        )

        print(
            "  premise:"
        )

        print(
            " ",
            (
                result.get(
                    "provenance_text"
                )
                or
                result.get(
                    "text"
                )
            )
        )


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


    records = (
        payload[
            "records"
        ]
    )


    print(
        "Loading NLI model:",
        NLI_MODEL_NAME
    )


    model = (
        CrossEncoder(
            NLI_MODEL_NAME
        )
    )


    labels = (
        get_labels(
            model
        )
    )


    print(
        "Runtime labels:",
        labels
    )


    if set(
        labels
    ) != {
        "contradiction",
        "entailment",
        "neutral",
    }:

        raise RuntimeError(
            "Unexpected NLI label mapping: "
            f"{labels}"
        )


    run_sanity_check(
        model,
        labels,
    )


    output = []


    summary = {
        "unsupported_claims":
            0,

        "rescued_by_provenance":
            0,

        "rescued_by_cross_citation_pair":
            0,

        "still_not_entailed":
            0,
    }


    for record in records:

        generation = (
            record.get(
                "generation"
            )
        )


        if generation is None:

            continue


        attempts = (
            record[
                "evidence_attempts"
            ]
        )


        final_context = (
            attempts[
                -1
            ][
                "context"
            ]
        )


        context_items = (
            final_context[
                "items"
            ]
        )


        print(
            "\n\n"
            +
            "=" * 100
        )

        print(
            record[
                "target"
            ].upper()
        )

        print(
            "=" * 100
        )

        print(
            record[
                "question"
            ]
        )


        claim_outputs = []


        for (
            index,
            claim_record,
        ) in enumerate(
            generation[
                "claims"
            ],
            start=1,
        ):

            claim = (
                claim_record[
                    "claim"
                ]
            )


            print(
                "\n"
                +
                "-" * 100
            )

            print(
                f"CLAIM {index}"
            )

            print(
                claim
            )

            print(
                "Production supported:",
                claim_record[
                    "supported"
                ]
            )

            print(
                "Production label:",
                claim_record[
                    "label"
                ]
            )

            print(
                "Production entailment:",
                claim_record[
                    "entailment_score"
                ]
            )


            modes = (
                diagnose_claim(
                    model=
                        model,
                    labels=
                        labels,
                    claim=
                        claim,
                    context_items=
                        context_items,
                )
            )


            print_mode(
                "A — PLAIN",
                modes[
                    "plain"
                ],
            )


            print_mode(
                "B — PROVENANCE",
                modes[
                    "provenance"
                ],
            )


            print_mode(
                "C — CROSS-CITATION PAIR",
                modes[
                    "cross_citation_pair"
                ],
            )


            if not claim_record[
                "supported"
            ]:

                summary[
                    "unsupported_claims"
                ] += 1


                provenance_rescued = (
                    modes[
                        "provenance"
                    ]
                    is not None
                    and
                    modes[
                        "provenance"
                    ][
                        "label"
                    ]
                    ==
                    "entailment"
                )


                pair_rescued = (
                    modes[
                        "cross_citation_pair"
                    ]
                    is not None
                    and
                    modes[
                        "cross_citation_pair"
                    ][
                        "label"
                    ]
                    ==
                    "entailment"
                )


                if provenance_rescued:

                    summary[
                        "rescued_by_provenance"
                    ] += 1


                if (
                    not provenance_rescued
                    and
                    pair_rescued
                ):

                    summary[
                        "rescued_by_cross_citation_pair"
                    ] += 1


                if (
                    not provenance_rescued
                    and
                    not pair_rescued
                ):

                    summary[
                        "still_not_entailed"
                    ] += 1


            claim_outputs.append(
                {
                    "claim":
                        claim,

                    "production":
                        claim_record,

                    "modes":
                        modes,
                }
            )


        output.append(
            {
                "target":
                    record[
                        "target"
                    ],

                "question":
                    record[
                        "question"
                    ],

                "gold_answer":
                    record.get(
                        "gold_answer"
                    ),

                "claims":
                    claim_outputs,
            }
        )


    print(
        "\n\n"
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