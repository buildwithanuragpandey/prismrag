import hashlib
import json
from pathlib import Path


FROZEN_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)

DEV_PATH = Path(
    "evaluation/datasets/router_dev_control_180.json"
)

CORPUS_PATH = Path(
    "data/processed/processed_corpus.json"
)

OUTPUT_PATH = Path(
    "evaluation/datasets/router_test_control_180.json"
)


SIMPLE_COUNT = 60
MULTIHOP_COUNT = 60
COMPLEX_COUNT = 60

#
# Different from dev selection seed.
#
SEED = 20260829


# ============================================================
# Utilities
# ============================================================

def stable_hash(
    text: str
) -> str:

    return hashlib.sha256(
        f"{SEED}:{text}".encode(
            "utf-8"
        )
    ).hexdigest()


def deterministic_sample(
    items: list[dict],
    count: int,
    key_name: str
):

    ranked = sorted(
        items,
        key=lambda item: stable_hash(
            str(
                item[
                    key_name
                ]
            )
        )
    )

    if len(ranked) < count:

        raise ValueError(
            f"Need {count} examples, "
            f"but only {len(ranked)} are available."
        )

    return ranked[:count]


def load_json(
    path: Path
):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ============================================================
# Load unique corpus documents
# ============================================================

def load_unique_documents():

    chunks = load_json(
        CORPUS_PATH
    )


    if isinstance(
        chunks,
        dict
    ):

        chunks = chunks.get(
            "chunks",
            []
        )


    documents = {}


    for chunk in chunks:

        document_id = chunk.get(
            "document_id"
        )


        if not document_id:

            continue


        if document_id in documents:

            continue


        metadata = (
            chunk.get(
                "metadata"
            )
            or {}
        )


        title = metadata.get(
            "title"
        )


        if not title:

            continue


        documents[
            document_id
        ] = {
            "document_id":
                document_id,

            "title":
                title,

            "author":
                metadata.get(
                    "author"
                ),

            "source":
                metadata.get(
                    "source"
                ),

            "published_at":
                metadata.get(
                    "published_at"
                )
        }


    return list(
        documents.values()
    )


# ============================================================
# Find examples already used in dev
# ============================================================

def build_dev_exclusions(
    dev_examples: list[dict]
):

    used_source_eval_ids = set()

    used_simple_document_ids = set()

    used_questions = set()


    for example in dev_examples:

        used_questions.add(
            example[
                "question"
            ]
        )


        source_eval_id = (
            example.get(
                "source_eval_id"
            )
        )


        if source_eval_id:

            used_source_eval_ids.add(
                source_eval_id
            )


        if (
            example[
                "gold_query_type"
            ]
            ==
            "simple"
        ):

            for document_id in (
                example.get(
                    "gold_document_ids",
                    []
                )
            ):

                used_simple_document_ids.add(
                    document_id
                )


    return {
        "source_eval_ids":
            used_source_eval_ids,

        "simple_document_ids":
            used_simple_document_ids,

        "questions":
            used_questions
    }


# ============================================================
# Held-out simple controls
# ============================================================

def build_simple_examples(
    documents: list[dict],
    exclusions: dict
):

    candidates = [

        document

        for document in documents

        if (
            document[
                "document_id"
            ]
            not in
            exclusions[
                "simple_document_ids"
            ]
        )
    ]


    selected = deterministic_sample(
        candidates,
        SIMPLE_COUNT,
        "document_id"
    )


    #
    # Important:
    #
    # These phrasings are DIFFERENT
    # from the dev templates.
    #

    templates = [

        'Who is the author of "{title}"?',

        'What is the publication date of "{title}"?',

        'Where was "{title}" published?'
    ]


    examples = []


    for index, document in enumerate(
        selected
    ):

        template = templates[
            index
            %
            len(
                templates
            )
        ]


        question = template.format(
            title=document[
                "title"
            ]
        )


        examples.append(
            {
                "id":
                    f"router_test_simple_{index:03d}",

                "question":
                    question,

                "source":
                    "synthetic_single_document_control_heldout",

                "gold_query_type":
                    "simple",

                "gold_retrieval_strategy":
                    "dense",

                "gold_rerank":
                    False,

                "gold_mmr":
                    False,

                "gold_document_ids":
                    [
                        document[
                            "document_id"
                        ]
                    ],

                "metadata":
                    {
                        "title":
                            document[
                                "title"
                            ],

                        "author":
                            document[
                                "author"
                            ],

                        "source":
                            document[
                                "source"
                            ],

                        "published_at":
                            document[
                                "published_at"
                            ]
                    }
            }
        )


    return examples


# ============================================================
# Held-out multihop
# ============================================================

def build_multihop_examples(
    frozen: list[dict],
    exclusions: dict
):

    candidates = [

        example

        for example in frozen

        if (
            example[
                "question_type"
            ]
            ==
            "inference_query"
        )

        and

        (
            example[
                "id"
            ]
            not in
            exclusions[
                "source_eval_ids"
            ]
        )

        and

        (
            example[
                "question"
            ]
            not in
            exclusions[
                "questions"
            ]
        )
    ]


    selected = deterministic_sample(
        candidates,
        MULTIHOP_COUNT,
        "id"
    )


    return [

        {
            "id":
                f"router_test_multihop_{index:03d}",

            "question":
                example[
                    "question"
                ],

            "source":
                "multihoprag_inference_heldout",

            "source_eval_id":
                example[
                    "id"
                ],

            "gold_query_type":
                "multihop",

            "gold_retrieval_strategy":
                "hybrid",

            "gold_rerank":
                True,

            "gold_mmr":
                True,

            "gold_document_ids":
                example[
                    "evidence_document_ids"
                ]
        }

        for index, example in enumerate(
            selected
        )
    ]


# ============================================================
# Held-out complex
# ============================================================

def build_complex_examples(
    frozen: list[dict],
    exclusions: dict
):

    comparison = [

        example

        for example in frozen

        if (
            example[
                "question_type"
            ]
            ==
            "comparison_query"
        )

        and

        (
            example[
                "id"
            ]
            not in
            exclusions[
                "source_eval_ids"
            ]
        )

        and

        (
            example[
                "question"
            ]
            not in
            exclusions[
                "questions"
            ]
        )
    ]


    temporal = [

        example

        for example in frozen

        if (
            example[
                "question_type"
            ]
            ==
            "temporal_query"
        )

        and

        (
            example[
                "id"
            ]
            not in
            exclusions[
                "source_eval_ids"
            ]
        )

        and

        (
            example[
                "question"
            ]
            not in
            exclusions[
                "questions"
            ]
        )
    ]


    selected_comparison = deterministic_sample(
        comparison,
        COMPLEX_COUNT // 2,
        "id"
    )


    selected_temporal = deterministic_sample(
        temporal,
        COMPLEX_COUNT // 2,
        "id"
    )


    selected = (
        selected_comparison
        +
        selected_temporal
    )


    return [

        {
            "id":
                f"router_test_complex_{index:03d}",

            "question":
                example[
                    "question"
                ],

            "source":
                (
                    example[
                        "question_type"
                    ]
                    +
                    "_heldout"
                ),

            "source_eval_id":
                example[
                    "id"
                ],

            "gold_query_type":
                "complex",

            "gold_retrieval_strategy":
                "hybrid",

            "gold_rerank":
                True,

            "gold_mmr":
                True,

            "gold_document_ids":
                example[
                    "evidence_document_ids"
                ]
        }

        for index, example in enumerate(
            selected
        )
    ]


# ============================================================
# Validation
# ============================================================

def validate(
    examples: list[dict],
    dev_examples: list[dict]
):

    assert len(
        examples
    ) == 180


    ids = [
        example[
            "id"
        ]
        for example in examples
    ]


    questions = [
        example[
            "question"
        ]
        for example in examples
    ]


    assert len(
        ids
    ) == len(
        set(ids)
    )


    assert len(
        questions
    ) == len(
        set(questions)
    )


    dev_questions = {
        example[
            "question"
        ]
        for example in dev_examples
    }


    overlap = (
        set(
            questions
        )
        &
        dev_questions
    )


    if overlap:

        raise ValueError(
            "Router test contains dev questions: "
            f"{list(overlap)[:5]}"
        )


    distribution = {}


    for example in examples:

        label = example[
            "gold_query_type"
        ]

        distribution[
            label
        ] = (
            distribution.get(
                label,
                0
            )
            +
            1
        )


    expected = {
        "simple": 60,
        "multihop": 60,
        "complex": 60
    }


    assert distribution == expected


    print(
        "\n===== HELD-OUT ROUTER TEST SET ====="
    )

    print(
        "Total:",
        len(
            examples
        )
    )

    print(
        "Distribution:",
        distribution
    )

    print(
        "Question overlap with dev:",
        len(
            overlap
        )
    )

    print(
        "Validation: OK"
    )


# ============================================================
# Main
# ============================================================

def main():

    frozen = load_json(
        FROZEN_PATH
    )

    dev = load_json(
        DEV_PATH
    )

    documents = (
        load_unique_documents()
    )


    exclusions = (
        build_dev_exclusions(
            dev
        )
    )


    print(
        "Frozen examples:",
        len(
            frozen
        )
    )

    print(
        "Dev examples:",
        len(
            dev
        )
    )

    print(
        "Unique corpus documents:",
        len(
            documents
        )
    )

    print(
        "Excluded source eval IDs:",
        len(
            exclusions[
                "source_eval_ids"
            ]
        )
    )

    print(
        "Excluded simple documents:",
        len(
            exclusions[
                "simple_document_ids"
            ]
        )
    )


    simple = build_simple_examples(
        documents,
        exclusions
    )


    multihop = build_multihop_examples(
        frozen,
        exclusions
    )


    complex_examples = build_complex_examples(
        frozen,
        exclusions
    )


    examples = (
        simple
        +
        multihop
        +
        complex_examples
    )


    validate(
        examples,
        dev
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            examples,
            file,
            ensure_ascii=False,
            indent=2
        )


    print(
        "Saved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()