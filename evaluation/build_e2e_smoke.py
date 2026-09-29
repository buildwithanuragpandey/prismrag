import json
import random

from pathlib import Path


SOURCE_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)

OUTPUT_PATH = Path(
    "evaluation/datasets/frozen_e2e_smoke_20.json"
)

MANIFEST_PATH = Path(
    "evaluation/datasets/frozen_e2e_smoke_20_manifest.json"
)


SEED = 26


TARGET_COUNTS = {
    "inference_query": 5,
    "comparison_query": 5,
    "temporal_query": 5,
    "null_query": 5,
}


def load_examples():

    with open(
        SOURCE_PATH,
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
        "Could not locate examples "
        "inside frozen_eval_500.json"
    )


def stable_sort_key(
    example,
):

    return (
        str(
            example.get(
                "id",
                ""
            )
        ),
        str(
            example.get(
                "question",
                ""
            )
        ),
    )


def main():

    examples = (
        load_examples()
    )


    rng = (
        random.Random(
            SEED
        )
    )


    selected = []

    distribution = {}


    for (
        question_type,
        target_count,
    ) in TARGET_COUNTS.items():

        candidates = [

            example

            for example
            in examples

            if (
                example.get(
                    "question_type"
                )
                ==
                question_type
            )
        ]


        candidates = sorted(
            candidates,
            key=
                stable_sort_key,
        )


        if (
            len(
                candidates
            )
            <
            target_count
        ):

            raise ValueError(
                (
                    f"Not enough examples for "
                    f"{question_type}: "
                    f"{len(candidates)}"
                )
            )


        chosen = (
            rng.sample(
                candidates,
                target_count,
            )
        )


        chosen = sorted(
            chosen,
            key=
                stable_sort_key,
        )


        selected.extend(
            chosen
        )


        distribution[
            question_type
        ] = len(
            chosen
        )


    # ========================================================
    # Keep output grouped deterministically by question type
    # ========================================================

    type_order = {
        "inference_query": 0,
        "comparison_query": 1,
        "temporal_query": 2,
        "null_query": 3,
    }


    selected = sorted(
        selected,
        key=lambda example: (
            type_order.get(
                example.get(
                    "question_type"
                ),
                999,
            ),
            stable_sort_key(
                example
            ),
        ),
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
            selected,
            file,
            indent=2,
            ensure_ascii=False,
        )


    manifest = {
        "source":
            str(
                SOURCE_PATH
            ),

        "output":
            str(
                OUTPUT_PATH
            ),

        "seed":
            SEED,

        "total":
            len(
                selected
            ),

        "distribution":
            distribution,

        "ids": [
            example.get(
                "id"
            )
            for example
            in selected
        ],
    }


    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


    print(
        "\nFrozen E2E smoke set created."
    )


    print(
        "Total:",
        len(
            selected
        )
    )


    print(
        "Distribution:"
    )


    for (
        question_type,
        count,
    ) in distribution.items():

        print(
            f"  {question_type}: "
            f"{count}"
        )


    print(
        "\nDataset:"
    )

    print(
        OUTPUT_PATH
    )


    print(
        "\nManifest:"
    )

    print(
        MANIFEST_PATH
    )


if __name__ == "__main__":

    main()