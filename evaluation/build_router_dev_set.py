import hashlib
import json
from pathlib import Path


FROZEN_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)

CORPUS_PATH = Path(
    "data/processed/processed_corpus.json"
)

OUTPUT_PATH = Path(
    "evaluation/datasets/router_dev_control_180.json"
)


SIMPLE_COUNT = 60
MULTIHOP_COUNT = 60
COMPLEX_COUNT = 60

SEED = 42


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

    return ranked[:count]


# ============================================================
# Load frozen MultiHopRAG
# ============================================================

def load_frozen():

    with open(
        FROZEN_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ============================================================
# Load unique documents from chunk corpus
# ============================================================

def load_unique_documents():

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(
            file
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

        document_id = (
            chunk.get(
                "document_id"
            )
        )


        if not document_id:

            continue


        if (
            document_id
            in documents
        ):

            continue


        metadata = (
            chunk.get(
                "metadata"
            )
            or {}
        )


        title = (
            metadata.get(
                "title"
            )
        )


        author = (
            metadata.get(
                "author"
            )
        )


        source = (
            metadata.get(
                "source"
            )
        )


        published_at = (
            metadata.get(
                "published_at"
            )
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
                author,

            "source":
                source,

            "published_at":
                published_at
        }


    return list(
        documents.values()
    )


# ============================================================
# Simple controls
# ============================================================

def build_simple_examples(
    documents: list[dict]
):

    selected_documents = (
        deterministic_sample(
            documents,
            SIMPLE_COUNT,
            "document_id"
        )
    )


    templates = [
        "Who wrote the article \"{title}\"?",
        "When was the article \"{title}\" published?",
        "Which source published the article \"{title}\"?"
    ]


    examples = []


    for index, document in enumerate(
        selected_documents
    ):

        template = (
            templates[
                index
                %
                len(
                    templates
                )
            ]
        )


        question = (
            template.format(
                title=document[
                    "title"
                ]
            )
        )


        examples.append(
            {
                "id":
                    f"router_simple_{index:03d}",

                "question":
                    question,

                "source":
                    "synthetic_single_document_control",

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
# MultiHop controls
# ============================================================

def build_multihop_examples(
    frozen: list[dict]
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
    ]


    selected = (
        deterministic_sample(
            candidates,
            MULTIHOP_COUNT,
            "id"
        )
    )


    return [

        {
            "id":
                f"router_multihop_{index:03d}",

            "question":
                example[
                    "question"
                ],

            "source":
                "multihoprag_inference",

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

        for index, example
        in enumerate(
            selected
        )
    ]


# ============================================================
# Complex controls
# ============================================================

def build_complex_examples(
    frozen: list[dict]
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
    ]


    selected_comparison = (
        deterministic_sample(
            comparison,
            COMPLEX_COUNT // 2,
            "id"
        )
    )


    selected_temporal = (
        deterministic_sample(
            temporal,
            COMPLEX_COUNT // 2,
            "id"
        )
    )


    selected = (
        selected_comparison
        +
        selected_temporal
    )


    return [

        {
            "id":
                f"router_complex_{index:03d}",

            "question":
                example[
                    "question"
                ],

            "source":
                example[
                    "question_type"
                ],

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

        for index, example
        in enumerate(
            selected
        )
    ]


# ============================================================
# Validation
# ============================================================

def validate(
    examples: list[dict]
):

    assert (
        len(examples)
        ==
        180
    )


    ids = [
        example["id"]
        for example in examples
    ]


    assert (
        len(ids)
        ==
        len(set(ids))
    )


    distribution = {}


    for example in examples:

        label = (
            example[
                "gold_query_type"
            ]
        )

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


    assert (
        distribution[
            "simple"
        ]
        ==
        60
    )


    assert (
        distribution[
            "multihop"
        ]
        ==
        60
    )


    assert (
        distribution[
            "complex"
        ]
        ==
        60
    )


    print(
        "\n===== ROUTER DEV SET ====="
    )

    print(
        "Total:",
        len(examples)
    )

    print(
        "Distribution:",
        distribution
    )

    print(
        "Validation: OK"
    )


# ============================================================
# Main
# ============================================================

def main():

    frozen = (
        load_frozen()
    )


    documents = (
        load_unique_documents()
    )


    print(
        "Unique corpus documents:",
        len(documents)
    )


    simple = (
        build_simple_examples(
            documents
        )
    )


    multihop = (
        build_multihop_examples(
            frozen
        )
    )


    complex_examples = (
        build_complex_examples(
            frozen
        )
    )


    examples = (
        simple
        +
        multihop
        +
        complex_examples
    )


    validate(
        examples
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