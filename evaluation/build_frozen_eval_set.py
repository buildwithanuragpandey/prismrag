import hashlib
import json
import random

from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "yixuantt/MultiHopRAG"
DATASET_CONFIG = "MultiHopRAG"

#
# Pin the dataset revision for reproducibility.
#

DATASET_REVISION = (
    "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
)


PROCESSED_CORPUS_PATH = Path(
    "data/processed/processed_corpus.json"
)

OUTPUT_DIR = Path(
    "evaluation/datasets"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "frozen_eval_500.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "frozen_eval_500_manifest.json"
)


SEED = 42


QUOTAS = {
    "inference_query": 140,
    "comparison_query": 140,
    "temporal_query": 140,
    "null_query": 80,
}


# ============================================================
# Utilities
# ============================================================

def normalize_text(
    value: str | None
) -> str:

    if value is None:
        return ""

    return " ".join(
        value
        .strip()
        .lower()
        .split()
    )


def stable_hash(
    text: str,
    seed: int = SEED
) -> str:

    payload = (
        f"{seed}:{text}"
        .encode("utf-8")
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def make_example_id(
    question: str
) -> str:

    digest = hashlib.sha1(
        question.encode(
            "utf-8"
        )
    ).hexdigest()[:12]

    return f"eval_{digest}"


# ============================================================
# Load processed corpus
# ============================================================

def load_processed_chunks() -> list[dict]:

    with open(
        PROCESSED_CORPUS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )


    #
    # Support both:
    #
    # [...]
    #
    # and:
    #
    # {"chunks": [...]}
    #

    if isinstance(
        data,
        list
    ):

        return data


    if isinstance(
        data,
        dict
    ):

        if (
            "chunks"
            in data
        ):

            return data[
                "chunks"
            ]


    raise ValueError(
        "Unsupported processed corpus format."
    )


# ============================================================
# Build evidence → document_id indexes
# ============================================================

def build_document_indexes(
    chunks: list[dict]
):

    url_to_document_id = {}

    title_to_document_id = {}


    for chunk in chunks:

        document_id = (
            chunk.get(
                "document_id"
            )
        )


        if not document_id:

            continue


        metadata = (
            chunk.get(
                "metadata"
            )
            or
            {}
        )


        #
        # Support both nested metadata
        # and flattened chunk structures.
        #

        url = (
            metadata.get("url")
            or
            chunk.get("url")
        )


        title = (
            metadata.get("title")
            or
            chunk.get("title")
        )


        if url:

            normalized_url = (
                normalize_text(
                    url
                )
            )

            existing = (
                url_to_document_id
                .get(
                    normalized_url
                )
            )


            if (
                existing is not None
                and
                existing != document_id
            ):

                raise ValueError(
                    "URL maps to multiple "
                    f"document IDs: {url}"
                )


            url_to_document_id[
                normalized_url
            ] = document_id


        if title:

            normalized_title = (
                normalize_text(
                    title
                )
            )

            existing = (
                title_to_document_id
                .get(
                    normalized_title
                )
            )


            #
            # Duplicate titles may theoretically
            # exist. In that case we simply avoid
            # using title as an ambiguous fallback.
            #

            if existing is None:

                title_to_document_id[
                    normalized_title
                ] = document_id

            elif (
                existing
                !=
                document_id
            ):

                title_to_document_id[
                    normalized_title
                ] = None


    return (
        url_to_document_id,
        title_to_document_id
    )


# ============================================================
# Evidence mapping
# ============================================================

def map_evidence_item(
    evidence: Any,
    url_index: dict,
    title_index: dict
) -> dict:


    if not isinstance(
        evidence,
        dict
    ):

        raise ValueError(
            "Expected evidence item "
            "to be a dictionary, got: "
            f"{type(evidence)}"
        )


    url = evidence.get(
        "url"
    )

    title = evidence.get(
        "title"
    )

    fact = (
        evidence.get("fact")
        or
        evidence.get("text")
        or
        evidence.get("evidence")
        or
        ""
    )


    document_id = None


    # --------------------------------------------------------
    # Primary mapping:
    # URL
    # --------------------------------------------------------

    if url:

        document_id = (
            url_index.get(
                normalize_text(
                    url
                )
            )
        )


    # --------------------------------------------------------
    # Fallback:
    # title
    # --------------------------------------------------------

    if (
        document_id is None
        and
        title
    ):

        document_id = (
            title_index.get(
                normalize_text(
                    title
                )
            )
        )


    return {

        "document_id":
            document_id,

        "fact":
            fact,

        "title":
            title,

        "url":
            url
    }


# ============================================================
# Convert dataset row
# ============================================================

def convert_row(
    row: dict,
    source_index: int,
    url_index: dict,
    title_index: dict
) -> dict:


    question = (
        row[
            "query"
        ]
        .strip()
    )


    question_type = (
        row[
            "question_type"
        ]
    )


    raw_evidence = (
        row.get(
            "evidence_list"
        )
        or
        []
    )


    mapped_evidence = [

        map_evidence_item(
            evidence=item,
            url_index=url_index,
            title_index=title_index
        )

        for item in raw_evidence

    ]


    evidence_document_ids = list(
        dict.fromkeys(

            item[
                "document_id"
            ]

            for item
            in mapped_evidence

            if item[
                "document_id"
            ]
            is not None
        )
    )


    is_answerable = (
        question_type
        !=
        "null_query"
    )


    return {

        "id":
            make_example_id(
                question
            ),

        "source_index":
            source_index,

        "question":
            question,

        "answer":
            row.get(
                "answer"
            ),

        "question_type":
            question_type,

        "is_answerable":
            is_answerable,

        "evidence":
            mapped_evidence,

        "evidence_document_ids":
            evidence_document_ids,

        #
        # Used only to deterministically
        # select the frozen subset.
        #
        # Removed before saving.
        #

        "_selection_hash":
            stable_hash(
                question
            )
    }


# ============================================================
# Validate mapping BEFORE sampling
# ============================================================

def validate_mapping(
    examples: list[dict]
):

    total_evidence = 0

    mapped_evidence = 0

    unmapped = []


    for example in examples:

        for evidence in (
            example[
                "evidence"
            ]
        ):

            total_evidence += 1


            if (
                evidence[
                    "document_id"
                ]
                is not None
            ):

                mapped_evidence += 1

            else:

                unmapped.append(
                    {
                        "id":
                            example["id"],

                        "question":
                            example[
                                "question"
                            ],

                        "title":
                            evidence[
                                "title"
                            ],

                        "url":
                            evidence[
                                "url"
                            ]
                    }
                )


    coverage = (

        mapped_evidence
        /
        total_evidence

        if total_evidence

        else 1.0
    )


    print(
        "\n"
        "===== EVIDENCE MAPPING ====="
    )

    print(
        "Total evidence items:",
        total_evidence
    )

    print(
        "Mapped:",
        mapped_evidence
    )

    print(
        "Unmapped:",
        len(
            unmapped
        )
    )

    print(
        "Coverage:",
        round(
            coverage,
            4
        )
    )


    if unmapped:

        print(
            "\nFirst unmapped examples:"
        )

        for item in (
            unmapped[:10]
        ):

            print(
                item
            )


        raise ValueError(
            "Evidence mapping is not "
            "100%. Do not build frozen "
            "evaluation set yet."
        )


# ============================================================
# Deterministic sampling
# ============================================================

def build_frozen_subset(
    examples: list[dict]
) -> list[dict]:


    groups = {}


    for example in examples:

        question_type = (
            example[
                "question_type"
            ]
        )

        groups.setdefault(
            question_type,
            []
        ).append(
            example
        )


    selected = []


    for (
        question_type,
        quota
    ) in QUOTAS.items():


        candidates = (
            groups.get(
                question_type,
                []
            )
        )


        print(
            f"{question_type}: "
            f"{len(candidates)} available"
        )


        if (
            len(candidates)
            <
            quota
        ):

            raise ValueError(
                f"Not enough "
                f"{question_type} examples. "
                f"Need {quota}, "
                f"found {len(candidates)}."
            )


        #
        # Stable selection independent
        # of source row ordering.
        #

        ranked = sorted(

            candidates,

            key=lambda item: (
                item[
                    "_selection_hash"
                ],
                item[
                    "id"
                ]
            )
        )


        selected.extend(
            ranked[
                :quota
            ]
        )


    #
    # Deterministic final shuffle so
    # question types are mixed.
    #

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        selected
    )


    for example in selected:

        example.pop(
            "_selection_hash",
            None
        )


    return selected


# ============================================================
# Validate final frozen set
# ============================================================

def validate_frozen_set(
    examples: list[dict]
):


    assert (
        len(examples)
        ==
        sum(
            QUOTAS.values()
        )
    )


    ids = [
        item["id"]
        for item in examples
    ]


    questions = [
        item["question"]
        for item in examples
    ]


    assert (
        len(ids)
        ==
        len(set(ids))
    )


    assert (
        len(questions)
        ==
        len(set(questions))
    )


    distribution = Counter(

        item[
            "question_type"
        ]

        for item in examples
    )


    assert (
        distribution
        ==
        Counter(
            QUOTAS
        )
    )


    for example in examples:

        if (
            example[
                "is_answerable"
            ]
        ):

            assert (
                len(
                    example[
                        "evidence_document_ids"
                    ]
                )
                >
                0
            )


        else:

            assert (
                example[
                    "question_type"
                ]
                ==
                "null_query"
            )


    print(
        "\n"
        "===== FROZEN SET VALIDATION ====="
    )

    print(
        "Total:",
        len(
            examples
        )
    )

    print(
        "Unique IDs:",
        len(
            set(ids)
        )
    )

    print(
        "Unique questions:",
        len(
            set(questions)
        )
    )

    print(
        "Distribution:",
        dict(
            distribution
        )
    )

    print(
        "Frozen set validation: OK"
    )


# ============================================================
# Save
# ============================================================

def save_outputs(
    examples: list[dict],
    full_dataset_size: int
):

    OUTPUT_DIR.mkdir(
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


    manifest = {

        "dataset_name":
            DATASET_NAME,

        "dataset_config":
            DATASET_CONFIG,

        "dataset_revision":
            DATASET_REVISION,

        "source_dataset_size":
            full_dataset_size,

        "frozen_eval_size":
            len(examples),

        "seed":
            SEED,

        "quotas":
            QUOTAS,

        "processed_corpus_path":
            str(
                PROCESSED_CORPUS_PATH
            ),

        "selection_method":
            (
                "SHA256(seed + question) "
                "rank within question_type, "
                "then deterministic shuffle"
            )
    }


    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2
        )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        MANIFEST_PATH
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading processed corpus..."
    )


    chunks = (
        load_processed_chunks()
    )


    print(
        "Processed chunks:",
        len(
            chunks
        )
    )


    (
        url_index,
        title_index
    ) = build_document_indexes(
        chunks
    )


    print(
        "Unique indexed URLs:",
        len(
            url_index
        )
    )

    print(
        "Unique indexed titles:",
        len(
            [
                value
                for value
                in title_index.values()
                if value is not None
            ]
        )
    )


    print(
        "\nLoading MultiHopRAG..."
    )


    dataset = load_dataset(

        DATASET_NAME,

        DATASET_CONFIG,

        split="train",

        revision=(
            DATASET_REVISION
        )
    )


    print(
        "QA examples:",
        len(
            dataset
        )
    )


    examples = [

        convert_row(

            row=dict(row),

            source_index=index,

            url_index=url_index,

            title_index=title_index

        )

        for index, row
        in enumerate(dataset)

    ]


    print(
        "\n"
        "===== FULL DATASET DISTRIBUTION ====="
    )


    distribution = Counter(

        item[
            "question_type"
        ]

        for item
        in examples
    )


    for key, value in sorted(
        distribution.items()
    ):

        print(
            f"{key}: {value}"
        )


    #
    # Critical:
    # Do not freeze anything if
    # evidence mapping is broken.
    #

    validate_mapping(
        examples
    )


    frozen = (
        build_frozen_subset(
            examples
        )
    )


    validate_frozen_set(
        frozen
    )


    save_outputs(

        examples=frozen,

        full_dataset_size=(
            len(dataset)
        )
    )


if __name__ == "__main__":

    main()