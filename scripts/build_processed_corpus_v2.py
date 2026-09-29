import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from datasets import load_dataset

from adaptive_agentic_rag.data.multihop_adapter import (
    convert_corpus_to_documents,
)

from adaptive_agentic_rag.processing.chunker import (
    chunk_document,
)

from adaptive_agentic_rag.processing.cleaner import (
    clean_document,
)


# ============================================================
# Experiment configuration
# ============================================================

DATASET_NAME = "yixuantt/MultiHopRAG"

DATASET_REVISION = (
    "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
)

OUTPUT_PATH = Path(
    "data/processed/processed_corpus_v2.json"
)

FROZEN_EVAL_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)


# ------------------------------------------------------------
# V1:
#
# chunk_size    = 2000 characters
# chunk_overlap = 200 characters
#
# V2-A:
#
# approximately half the chunk size.
#
# This experiment intentionally changes ONLY chunking.
# Embedding representation remains unchanged for now.
# ------------------------------------------------------------

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 100

MIN_CHUNK_WORDS = 20


# ============================================================
# Helpers
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


def percentile(
    values: list[int],
    percentile_value: float,
):

    if not values:

        return 0


    ordered = sorted(
        values
    )


    position = int(
        round(
            (
                len(ordered)
                - 1
            )
            *
            percentile_value
        )
    )


    return ordered[
        position
    ]


# ============================================================
# Evaluation integrity
# ============================================================

def check_evidence_integrity(
    chunks: list[dict],
):

    if not FROZEN_EVAL_PATH.exists():

        print(
            "\nFrozen evaluation set not found."
        )

        return


    with open(
        FROZEN_EVAL_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        examples = json.load(
            file
        )


    by_document = defaultdict(
        list
    )


    for chunk in chunks:

        by_document[
            chunk["document_id"]
        ].append(
            normalize(
                chunk["text"]
            )
        )


    total = 0

    mapped = 0

    missing = []


    for example in examples:

        for evidence in (
            example.get(
                "evidence",
                []
            )
        ):

            fact = normalize(
                evidence.get(
                    "fact",
                    ""
                )
            )


            if not fact:

                continue


            total += 1


            document_id = (
                evidence[
                    "document_id"
                ]
            )


            found = any(
                fact in chunk_text
                for chunk_text
                in by_document.get(
                    document_id,
                    [],
                )
            )


            if found:

                mapped += 1

            else:

                missing.append(
                    {
                        "example_id":
                            example["id"],

                        "document_id":
                            document_id,

                        "fact":
                            evidence.get(
                                "fact",
                                "",
                            ),
                    }
                )


    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "FROZEN EVIDENCE INTEGRITY"
    )

    print(
        "=" * 80
    )


    print(
        "Evidence facts:",
        total
    )

    print(
        "Exactly mapped:",
        mapped
    )


    mapping_rate = (
        mapped
        /
        total
        if total
        else 0.0
    )


    print(
        "Mapping rate:",
        round(
            mapping_rate,
            6,
        )
    )


    print(
        "Missing:",
        len(
            missing
        )
    )


    if missing:

        print(
            "\nFirst missing examples:"
        )


        for item in missing[:10]:

            print(
                "\n",
                item[
                    "example_id"
                ],
                item[
                    "document_id"
                ],
            )

            print(
                item[
                    "fact"
                ][:300]
            )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading MultiHopRAG corpus..."
    )


    corpus = load_dataset(
        DATASET_NAME,
        "corpus",
        revision=DATASET_REVISION,
    )["train"]


    print(
        "Converting documents..."
    )


    documents = (
        convert_corpus_to_documents(
            corpus
        )
    )


    print(
        "Documents:",
        len(
            documents
        )
    )


    print(
        "\nBuilding V2-A chunks..."
    )

    print(
        "chunk_size:",
        CHUNK_SIZE,
    )

    print(
        "chunk_overlap:",
        CHUNK_OVERLAP,
    )


    all_chunks = []


    for document in documents:

        cleaned_document = (
            clean_document(
                document
            )
        )


        chunks = chunk_document(
            cleaned_document,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            min_chunk_words=MIN_CHUNK_WORDS,
        )


        all_chunks.extend(
            chunks
        )


    output = [
        {
            "id":
                chunk.id,

            "document_id":
                chunk.document_id,

            "text":
                chunk.text,

            "metadata":
                {
                    **chunk.metadata,

                    "chunking_version":
                        "v2_a",

                    "chunk_size_chars":
                        CHUNK_SIZE,

                    "chunk_overlap_chars":
                        CHUNK_OVERLAP,
                },
        }

        for chunk in all_chunks
    ]


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
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # Corpus statistics
    # ========================================================

    word_counts = [
        len(
            chunk[
                "text"
            ].split()
        )
        for chunk
        in output
    ]


    chunks_per_document = defaultdict(
        int
    )


    for chunk in output:

        chunks_per_document[
            chunk[
                "document_id"
            ]
        ] += 1


    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "V2-A CORPUS STATISTICS"
    )

    print(
        "=" * 80
    )


    print(
        "Chunks:",
        len(
            output
        )
    )

    print(
        "Documents:",
        len(
            chunks_per_document
        )
    )

    print(
        "Mean words:",
        round(
            mean(
                word_counts
            ),
            2,
        )
    )

    print(
        "Median words:",
        round(
            median(
                word_counts
            ),
            2,
        )
    )

    print(
        "P90 words:",
        percentile(
            word_counts,
            0.90,
        )
    )

    print(
        "P95 words:",
        percentile(
            word_counts,
            0.95,
        )
    )

    print(
        "Min words:",
        min(
            word_counts
        )
    )

    print(
        "Max words:",
        max(
            word_counts
        )
    )


    print(
        "Mean chunks/document:",
        round(
            mean(
                chunks_per_document.values()
            ),
            2,
        )
    )


    # ========================================================
    # Critical safeguard:
    #
    # Smaller chunks must not destroy the frozen gold facts.
    # ========================================================

    check_evidence_integrity(
        output
    )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":

    main()