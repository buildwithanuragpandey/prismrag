import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from adaptive_agentic_rag.embeddings.model import (
    EmbeddingModel,
)


# ============================================================
# Configuration
# ============================================================

EVAL_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)

CORPUS_PATH = Path(
    "data/processed/processed_corpus.json"
)

OUTPUT_PATH = Path(
    "evaluation/results/"
    "semantic_dilution_v1.json"
)


QUESTION_TYPES = (
    "inference_query",
    "comparison_query",
    "temporal_query",
)

EXAMPLES_PER_TYPE = 40


# ============================================================
# Text helpers
# ============================================================

def normalize(
    text: str,
) -> str:

    text = (
        text
        or ""
    ).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def word_count(
    text: str,
) -> int:

    return len(
        (text or "").split()
    )


def split_paragraphs(
    text: str,
) -> list[str]:

    paragraphs = [
        item.strip()
        for item in re.split(
            r"\n\s*\n",
            text or "",
        )
        if item.strip()
    ]

    if not paragraphs:

        stripped = (
            text
            or ""
        ).strip()

        if stripped:

            return [
                stripped
            ]

    return paragraphs


# ============================================================
# Similarity
#
# Embeddings are already normalized by EmbeddingModel,
# therefore dot product == cosine similarity.
# ============================================================

def cosine(
    a,
    b,
) -> float:

    return float(
        np.dot(
            a,
            b,
        )
    )


# ============================================================
# Load data
# ============================================================

def load_json(
    path: Path,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# Deterministic stratified subset
# ============================================================

def select_examples(
    examples: list[dict],
) -> list[dict]:

    grouped = defaultdict(
        list
    )

    for example in examples:

        question_type = (
            example.get(
                "question_type"
            )
        )

        if (
            question_type
            in QUESTION_TYPES
        ):

            grouped[
                question_type
            ].append(
                example
            )


    selected = []


    for question_type in (
        QUESTION_TYPES
    ):

        items = sorted(
            grouped[
                question_type
            ],
            key=lambda item: (
                item["id"]
            ),
        )

        selected.extend(
            items[
                :EXAMPLES_PER_TYPE
            ]
        )


    return selected


# ============================================================
# Corpus lookup
# ============================================================

def build_document_lookup(
    chunks: list[dict],
) -> dict[str, list[dict]]:

    lookup = defaultdict(
        list
    )

    for chunk in chunks:

        document_id = (
            chunk.get(
                "document_id"
            )
        )

        if document_id:

            lookup[
                document_id
            ].append(
                chunk
            )


    for document_chunks in (
        lookup.values()
    ):

        document_chunks.sort(
            key=lambda item: (
                item.get(
                    "metadata",
                    {},
                ).get(
                    "chunk_index",
                    0,
                )
            )
        )


    return dict(
        lookup
    )


# ============================================================
# Exact gold fact -> chunk mapping
# ============================================================

def find_gold_chunk(
    evidence: dict,
    by_document: dict[
        str,
        list[dict]
    ],
):

    document_id = (
        evidence.get(
            "document_id"
        )
    )

    fact = (
        evidence.get(
            "fact",
            ""
        )
    )

    normalized_fact = (
        normalize(
            fact
        )
    )


    if not normalized_fact:

        return None


    candidates = (
        by_document.get(
            document_id,
            [],
        )
    )


    for chunk in candidates:

        normalized_chunk = (
            normalize(
                chunk.get(
                    "text",
                    ""
                )
            )
        )

        if (
            normalized_fact
            in normalized_chunk
        ):

            return chunk


    return None


# ============================================================
# Prepare evaluation records
# ============================================================

def build_records(
    examples: list[dict],
    by_document: dict[
        str,
        list[dict]
    ],
):

    records = []

    missing_fact_count = 0


    for example in examples:

        question = (
            example.get(
                "question",
                ""
            )
        )


        for evidence in (
            example.get(
                "evidence",
                []
            )
        ):

            chunk = (
                find_gold_chunk(
                    evidence,
                    by_document,
                )
            )


            if chunk is None:

                missing_fact_count += 1

                continue


            text = (
                chunk.get(
                    "text",
                    ""
                )
            )


            paragraphs = (
                split_paragraphs(
                    text
                )
            )


            metadata = (
                chunk.get(
                    "metadata",
                    {}
                )
                or {}
            )


            records.append(
                {
                    "example_id":
                        example["id"],

                    "question_type":
                        example.get(
                            "question_type",
                            "",
                        ),

                    "question":
                        question,

                    "document_id":
                        evidence.get(
                            "document_id"
                        ),

                    "chunk_id":
                        chunk.get(
                            "id"
                        ),

                    "fact":
                        evidence.get(
                            "fact",
                            "",
                        ),

                    "title":
                        metadata.get(
                            "title",
                            "",
                        ),

                    "source":
                        metadata.get(
                            "source",
                            "",
                        ),

                    "full_chunk":
                        text,

                    "paragraphs":
                        paragraphs,

                    "chunk_words":
                        word_count(
                            text
                        ),
                }
            )


    return (
        records,
        missing_fact_count,
    )


# ============================================================
# Batch embedding
# ============================================================

def evaluate_embeddings(
    records: list[dict],
):

    embedder = (
        EmbeddingModel()
    )


    print(
        "Encoding questions..."
    )


    query_embeddings = (
        embedder.encode_queries(
            [
                record[
                    "question"
                ]
                for record
                in records
            ],
            batch_size=32,
            show_progress_bar=True,
        )
    )


    print(
        "Encoding full chunks..."
    )


    full_embeddings = (
        embedder.encode_documents(
            [
                record[
                    "full_chunk"
                ]
                for record
                in records
            ],
            batch_size=32,
            show_progress_bar=True,
        )
    )


    # --------------------------------------------------------
    # Flatten paragraphs so they can be encoded in one batch.
    # --------------------------------------------------------

    paragraph_texts = []

    paragraph_ranges = []


    for record in records:

        start = len(
            paragraph_texts
        )


        paragraph_texts.extend(
            record[
                "paragraphs"
            ]
        )


        end = len(
            paragraph_texts
        )


        paragraph_ranges.append(
            (
                start,
                end,
            )
        )


    print(
        "Encoding paragraphs..."
    )


    paragraph_embeddings = (
        embedder.encode_documents(
            paragraph_texts,
            batch_size=32,
            show_progress_bar=True,
        )
    )


    # --------------------------------------------------------
    # Metadata + paragraph representations
    # --------------------------------------------------------

    metadata_paragraph_texts = []


    for record in records:

        title = (
            record[
                "title"
            ]
        )

        source = (
            record[
                "source"
            ]
        )


        for paragraph in (
            record[
                "paragraphs"
            ]
        ):

            representation = (
                f"{title}\n"
                f"{source}\n"
                f"{paragraph}"
            )

            metadata_paragraph_texts.append(
                representation
            )


    print(
        "Encoding title/source paragraphs..."
    )


    metadata_paragraph_embeddings = (
        embedder.encode_documents(
            metadata_paragraph_texts,
            batch_size=32,
            show_progress_bar=True,
        )
    )


    # ========================================================
    # Score each evidence item
    # ========================================================

    results = []


    for index, record in enumerate(
        records
    ):

        query_vector = (
            query_embeddings[
                index
            ]
        )


        full_score = cosine(
            query_vector,
            full_embeddings[
                index
            ],
        )


        start, end = (
            paragraph_ranges[
                index
            ]
        )


        best_paragraph_score = (
            -1.0
        )

        best_paragraph_index = (
            None
        )


        best_metadata_score = (
            -1.0
        )

        best_metadata_index = (
            None
        )


        for local_index, global_index in enumerate(
            range(
                start,
                end,
            )
        ):

            paragraph_score = (
                cosine(
                    query_vector,
                    paragraph_embeddings[
                        global_index
                    ],
                )
            )


            if (
                paragraph_score
                >
                best_paragraph_score
            ):

                best_paragraph_score = (
                    paragraph_score
                )

                best_paragraph_index = (
                    local_index
                )


            metadata_score = (
                cosine(
                    query_vector,
                    metadata_paragraph_embeddings[
                        global_index
                    ],
                )
            )


            if (
                metadata_score
                >
                best_metadata_score
            ):

                best_metadata_score = (
                    metadata_score
                )

                best_metadata_index = (
                    local_index
                )


        best_paragraph = (
            record[
                "paragraphs"
            ][
                best_paragraph_index
            ]
        )


        best_metadata_paragraph = (
            record[
                "paragraphs"
            ][
                best_metadata_index
            ]
        )


        results.append(
            {
                **{
                    key: value
                    for key, value
                    in record.items()
                    if key
                    not in {
                        "paragraphs",
                        "full_chunk",
                    }
                },

                "paragraph_count":
                    len(
                        record[
                            "paragraphs"
                        ]
                    ),

                "full_chunk_similarity":
                    full_score,

                "best_paragraph_similarity":
                    best_paragraph_score,

                "title_source_paragraph_similarity":
                    best_metadata_score,

                "paragraph_delta":
                    (
                        best_paragraph_score
                        -
                        full_score
                    ),

                "metadata_paragraph_delta":
                    (
                        best_metadata_score
                        -
                        full_score
                    ),

                "best_paragraph_words":
                    word_count(
                        best_paragraph
                    ),

                "best_paragraph_preview":
                    best_paragraph[
                        :500
                    ],

                "best_metadata_paragraph_preview":
                    best_metadata_paragraph[
                        :500
                    ],
            }
        )


    return results


# ============================================================
# Aggregate metrics
# ============================================================

def mean(
    values,
):

    values = list(
        values
    )

    if not values:

        return 0.0

    return float(
        np.mean(
            values
        )
    )


def percentage(
    values,
    predicate,
):

    values = list(
        values
    )

    if not values:

        return 0.0

    count = sum(
        1
        for value
        in values
        if predicate(
            value
        )
    )

    return (
        count
        /
        len(
            values
        )
    )


def summarize(
    results: list[dict],
):

    paragraph_deltas = [
        item[
            "paragraph_delta"
        ]
        for item
        in results
    ]

    metadata_deltas = [
        item[
            "metadata_paragraph_delta"
        ]
        for item
        in results
    ]


    summary = {
        "evidence_items":
            len(
                results
            ),

        "mean_chunk_words":
            mean(
                item[
                    "chunk_words"
                ]
                for item
                in results
            ),

        "mean_best_paragraph_words":
            mean(
                item[
                    "best_paragraph_words"
                ]
                for item
                in results
            ),

        "mean_full_chunk_similarity":
            mean(
                item[
                    "full_chunk_similarity"
                ]
                for item
                in results
            ),

        "mean_best_paragraph_similarity":
            mean(
                item[
                    "best_paragraph_similarity"
                ]
                for item
                in results
            ),

        "mean_title_source_paragraph_similarity":
            mean(
                item[
                    "title_source_paragraph_similarity"
                ]
                for item
                in results
            ),

        "mean_paragraph_delta":
            mean(
                paragraph_deltas
            ),

        "mean_metadata_paragraph_delta":
            mean(
                metadata_deltas
            ),

        "paragraph_improved_rate":
            percentage(
                paragraph_deltas,
                lambda value:
                    value > 0,
            ),

        "paragraph_delta_ge_003_rate":
            percentage(
                paragraph_deltas,
                lambda value:
                    value >= 0.03,
            ),

        "paragraph_delta_ge_005_rate":
            percentage(
                paragraph_deltas,
                lambda value:
                    value >= 0.05,
            ),

        "paragraph_delta_ge_010_rate":
            percentage(
                paragraph_deltas,
                lambda value:
                    value >= 0.10,
            ),

        "metadata_delta_ge_005_rate":
            percentage(
                metadata_deltas,
                lambda value:
                    value >= 0.05,
            ),
    }


    # --------------------------------------------------------
    # By question type
    # --------------------------------------------------------

    by_type = {}


    for question_type in (
        QUESTION_TYPES
    ):

        subset = [
            item
            for item
            in results
            if (
                item[
                    "question_type"
                ]
                ==
                question_type
            )
        ]


        if not subset:

            continue


        by_type[
            question_type
        ] = {
            "count":
                len(
                    subset
                ),

            "mean_full":
                mean(
                    item[
                        "full_chunk_similarity"
                    ]
                    for item
                    in subset
                ),

            "mean_paragraph":
                mean(
                    item[
                        "best_paragraph_similarity"
                    ]
                    for item
                    in subset
                ),

            "mean_metadata_paragraph":
                mean(
                    item[
                        "title_source_paragraph_similarity"
                    ]
                    for item
                    in subset
                ),

            "mean_paragraph_delta":
                mean(
                    item[
                        "paragraph_delta"
                    ]
                    for item
                    in subset
                ),

            "mean_metadata_delta":
                mean(
                    item[
                        "metadata_paragraph_delta"
                    ]
                    for item
                    in subset
                ),
        }


    summary[
        "by_question_type"
    ] = by_type


    # --------------------------------------------------------
    # By chunk length
    # --------------------------------------------------------

    bins = {
        "under_200":
            lambda words:
                words < 200,

        "200_to_299":
            lambda words:
                200 <= words < 300,

        "300_plus":
            lambda words:
                words >= 300,
    }


    by_chunk_length = {}


    for name, predicate in (
        bins.items()
    ):

        subset = [
            item
            for item
            in results
            if predicate(
                item[
                    "chunk_words"
                ]
            )
        ]


        if not subset:

            continue


        by_chunk_length[
            name
        ] = {
            "count":
                len(
                    subset
                ),

            "mean_full":
                mean(
                    item[
                        "full_chunk_similarity"
                    ]
                    for item
                    in subset
                ),

            "mean_paragraph":
                mean(
                    item[
                        "best_paragraph_similarity"
                    ]
                    for item
                    in subset
                ),

            "mean_delta":
                mean(
                    item[
                        "paragraph_delta"
                    ]
                    for item
                    in subset
                ),
        }


    summary[
        "by_chunk_length"
    ] = (
        by_chunk_length
    )


    return summary


# ============================================================
# Print report
# ============================================================

def print_report(
    selected_examples,
    records,
    missing_fact_count,
    results,
    summary,
):

    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "SEMANTIC DILUTION EVALUATION — V1"
    )

    print(
        "=" * 100
    )


    print(
        "Selected examples:",
        len(
            selected_examples
        )
    )

    print(
        "Mapped evidence items:",
        len(
            records
        )
    )

    print(
        "Evidence facts not exactly mapped:",
        missing_fact_count
    )


    print(
        "\nOVERALL"
    )

    print(
        "Mean chunk words:",
        round(
            summary[
                "mean_chunk_words"
            ],
            2,
        )
    )

    print(
        "Mean best paragraph words:",
        round(
            summary[
                "mean_best_paragraph_words"
            ],
            2,
        )
    )

    print(
        "Mean FULL similarity:",
        round(
            summary[
                "mean_full_chunk_similarity"
            ],
            4,
        )
    )

    print(
        "Mean PARAGRAPH similarity:",
        round(
            summary[
                "mean_best_paragraph_similarity"
            ],
            4,
        )
    )

    print(
        "Mean TITLE+SOURCE+PARAGRAPH:",
        round(
            summary[
                "mean_title_source_paragraph_similarity"
            ],
            4,
        )
    )

    print(
        "Mean paragraph delta:",
        round(
            summary[
                "mean_paragraph_delta"
            ],
            4,
        )
    )

    print(
        "Mean metadata paragraph delta:",
        round(
            summary[
                "mean_metadata_paragraph_delta"
            ],
            4,
        )
    )


    print(
        "\nIMPROVEMENT RATES"
    )

    print(
        "Paragraph > full:",
        round(
            summary[
                "paragraph_improved_rate"
            ],
            4,
        )
    )

    print(
        "Delta >= 0.03:",
        round(
            summary[
                "paragraph_delta_ge_003_rate"
            ],
            4,
        )
    )

    print(
        "Delta >= 0.05:",
        round(
            summary[
                "paragraph_delta_ge_005_rate"
            ],
            4,
        )
    )

    print(
        "Delta >= 0.10:",
        round(
            summary[
                "paragraph_delta_ge_010_rate"
            ],
            4,
        )
    )

    print(
        "Metadata delta >= 0.05:",
        round(
            summary[
                "metadata_delta_ge_005_rate"
            ],
            4,
        )
    )


    print(
        "\nBY QUESTION TYPE"
    )

    for question_type, values in (
        summary[
            "by_question_type"
        ].items()
    ):

        print(
            "\n",
            question_type,
        )

        for key, value in (
            values.items()
        ):

            if isinstance(
                value,
                float,
            ):

                value = round(
                    value,
                    4,
                )

            print(
                " ",
                key,
                "=",
                value,
            )


    print(
        "\nBY CHUNK LENGTH"
    )

    for name, values in (
        summary[
            "by_chunk_length"
        ].items()
    ):

        print(
            "\n",
            name,
        )

        for key, value in (
            values.items()
        ):

            if isinstance(
                value,
                float,
            ):

                value = round(
                    value,
                    4,
                )

            print(
                " ",
                key,
                "=",
                value,
            )


    # --------------------------------------------------------
    # Biggest gains
    # --------------------------------------------------------

    ranked = sorted(
        results,
        key=lambda item: (
            item[
                "paragraph_delta"
            ]
        ),
        reverse=True,
    )


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "TOP 10 PARAGRAPH GAINS"
    )

    print(
        "=" * 100
    )


    for item in ranked[:10]:

        print(
            "\n",
            item[
                "example_id"
            ],
            item[
                "chunk_id"
            ],
        )

        print(
            "Type:",
            item[
                "question_type"
            ]
        )

        print(
            "Chunk words:",
            item[
                "chunk_words"
            ]
        )

        print(
            "Full:",
            round(
                item[
                    "full_chunk_similarity"
                ],
                4,
            )
        )

        print(
            "Paragraph:",
            round(
                item[
                    "best_paragraph_similarity"
                ],
                4,
            )
        )

        print(
            "Delta:",
            round(
                item[
                    "paragraph_delta"
                ],
                4,
            )
        )

        print(
            "Paragraph preview:"
        )

        print(
            item[
                "best_paragraph_preview"
            ]
        )


    # --------------------------------------------------------
    # Biggest regressions
    # --------------------------------------------------------

    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "BOTTOM 10 PARAGRAPH DELTAS"
    )

    print(
        "=" * 100
    )


    for item in ranked[-10:]:

        print(
            "\n",
            item[
                "example_id"
            ],
            item[
                "chunk_id"
            ],
        )

        print(
            "Full:",
            round(
                item[
                    "full_chunk_similarity"
                ],
                4,
            )
        )

        print(
            "Paragraph:",
            round(
                item[
                    "best_paragraph_similarity"
                ],
                4,
            )
        )

        print(
            "Delta:",
            round(
                item[
                    "paragraph_delta"
                ],
                4,
            )
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading frozen evaluation set..."
    )

    examples = load_json(
        EVAL_PATH
    )


    print(
        "Loading processed corpus..."
    )

    chunks = load_json(
        CORPUS_PATH
    )


    selected_examples = (
        select_examples(
            examples
        )
    )


    print(
        "Selected examples:",
        len(
            selected_examples
        )
    )


    by_document = (
        build_document_lookup(
            chunks
        )
    )


    records, missing_fact_count = (
        build_records(
            selected_examples,
            by_document,
        )
    )


    print(
        "Exact-mapped evidence items:",
        len(
            records
        )
    )

    print(
        "Missing exact mappings:",
        missing_fact_count
    )


    if not records:

        raise RuntimeError(
            "No evidence records were mapped."
        )


    results = (
        evaluate_embeddings(
            records
        )
    )


    summary = (
        summarize(
            results
        )
    )


    print_report(
        selected_examples,
        records,
        missing_fact_count,
        results,
        summary,
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    payload = {
        "experiment":
            "semantic_dilution_v1",

        "purpose":
            (
                "Diagnostic comparison of current "
                "full chunks against paragraph-level "
                "representations. Gold evidence is "
                "used only for offline diagnosis."
            ),

        "examples_per_type":
            EXAMPLES_PER_TYPE,

        "question_types":
            list(
                QUESTION_TYPES
            ),

        "selected_example_count":
            len(
                selected_examples
            ),

        "missing_exact_fact_mappings":
            missing_fact_count,

        "summary":
            summary,

        "results":
            results,
    }


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()