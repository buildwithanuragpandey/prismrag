import json

from pathlib import Path

from adaptive_agentic_rag.retrieval.reranker import (
    BGEReranker,
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "relevance_safety_floor_calibration.json"
)


# ============================================================
# Manually audited query-claim pairs from previous controlled
# generation diagnostics.
#
# positive:
#   claim is useful/relevant evidence for answering query
#
# negative:
#   claim is supported somewhere, but is not useful evidence
#   for answering this specific query
#
# This is NOT the final answer benchmark.
# It is calibration for a catastrophic-low-relevance floor.
# ============================================================

PAIRS = [

    # ========================================================
    # POSITIVES
    # ========================================================

    {
        "id": "positive_taylor_connection",

        "label": 1,

        "query": (
            "Was the news about Taylor Swift's relationship "
            "with Travis Kelce inconsistent with the later "
            "report from The Independent - Life and Style "
            "on December 6, 2023?"
        ),

        "claim": (
            "Taylor Swift revealed her connection with "
            "Travis Kelce in July after he confessed "
            "on his podcast."
        ),
    },

    {
        "id": "positive_taylor_bracelet",

        "label": 1,

        "query": (
            "Was the news about Taylor Swift's relationship "
            "with Travis Kelce inconsistent with the later "
            "report from The Independent - Life and Style "
            "on December 6, 2023?"
        ),

        "claim": (
            "Fans spotted Travis Kelce wearing a friendship "
            "bracelet with Taylor Swift lyrics on it days "
            "before she attended his game."
        ),
    },

    {
        "id": "positive_altman_candor",

        "label": 1,

        "query": (
            "Did the Fortune article published later on the "
            "same day contradict the TechCrunch article from "
            "November 18, 2023, regarding the circumstances "
            "surrounding Sam Altman's departure from OpenAI?"
        ),

        "claim": (
            "Altman's perceived lack of candor."
        ),
    },

    {
        "id": "positive_mctominay",

        "label": 1,

        "query": (
            "Does the Sporting News article identify Scott "
            "McTominay as Manchester United's top scorer for "
            "the season, while the TalkSport article suggests "
            "Erling Haaland has the chance to become the "
            "overall top scorer in 2023?"
        ),

        "claim": (
            "Scott McTominay is identified as Manchester "
            "United's top scorer for the season according "
            "to the Sporting News article."
        ),
    },

    {
        "id": "positive_haaland",

        "label": 1,

        "query": (
            "Does the Sporting News article identify Scott "
            "McTominay as Manchester United's top scorer for "
            "the season, while the TalkSport article suggests "
            "Erling Haaland has the chance to become the "
            "overall top scorer in 2023?"
        ),

        "claim": (
            "Erling Haaland is mentioned as having the "
            "potential to become the overall top scorer "
            "in 2023 based on the TalkSport article."
        ),
    },


    # ========================================================
    # NEGATIVES
    # ========================================================

    {
        "id": "negative_nigeria_starnews",

        "label": 0,

        "query": (
            "Considering the economic forecast from a "
            "Bloomberg article and the agricultural "
            "developments discussed in a Reuters report, "
            "which country in West Africa, expected to see "
            "a significant growth in its GDP, also launched "
            "an initiative to become self-sufficient in rice "
            "production by 2025?"
        ),

        "claim": (
            "France backs African mobile video startup "
            "StarNews Mobile in a $3 million round."
        ),
    },

    {
        "id": "negative_sbf_trial_title",

        "label": 0,

        "query": (
            "Who is the individual whose legal team and the "
            "government's attorneys are presenting conflicting "
            "narratives in court, who acknowledged being aware "
            "of a significant financial discrepancy, and is "
            "accused of instructing a subordinate to use "
            "billions of customer funds to settle debts?"
        ),

        "claim": (
            "The FTX trial is bigger than "
            "Sam Bankman-Fried - The Verge."
        ),
    },

    {
        "id": "negative_valve_future",

        "label": 0,

        "query": (
            "Which company, covered by Engadget and Polygon, "
            "is set to release updated gaming hardware with "
            "over 300 improvements on November 16, emphasizing "
            "a singular performance target for developers?"
        ),

        "claim": (
            "Valve plans for successive generations of "
            "handhelds, indicating ongoing development efforts "
            "beyond the current Steam Deck."
        ),
    },
]


def score_pairs(
    reranker,
):

    records = []


    for pair in PAIRS:

        ranked = reranker.rerank(
            query=pair[
                "query"
            ],
            documents=[
                {
                    "id":
                        pair[
                            "id"
                        ],

                    "text":
                        pair[
                            "claim"
                        ],
                }
            ],
            top_k=1,
        )


        score = float(
            ranked[
                0
            ][
                "rerank_score"
            ]
        )


        records.append(
            {
                **pair,

                "score":
                    score,
            }
        )


    return records


def threshold_candidates(
    records,
):

    values = sorted(
        set(
            record[
                "score"
            ]

            for record
            in records
        )
    )


    candidates = [
        values[
            0
        ]
        -
        0.001
    ]


    for (
        left,
        right,
    ) in zip(
        values,
        values[
            1:
        ],
    ):

        candidates.append(
            (
                left
                +
                right
            )
            /
            2
        )


    candidates.append(
        values[
            -1
        ]
        +
        0.001
    )


    return candidates


def evaluate_threshold(
    records,
    threshold,
):

    positives = [
        record

        for record
        in records

        if record[
            "label"
        ]
        ==
        1
    ]


    negatives = [
        record

        for record
        in records

        if record[
            "label"
        ]
        ==
        0
    ]


    positive_kept = sum(
        1

        for record
        in positives

        if record[
            "score"
        ]
        >=
        threshold
    )


    negative_rejected = sum(
        1

        for record
        in negatives

        if record[
            "score"
        ]
        <
        threshold
    )


    return {
        "threshold":
            threshold,

        "positive_total":
            len(
                positives
            ),

        "positive_kept":
            positive_kept,

        "positive_keep_rate": (
            positive_kept
            /
            len(
                positives
            )
        ),

        "negative_total":
            len(
                negatives
            ),

        "negative_rejected":
            negative_rejected,

        "negative_reject_rate": (
            negative_rejected
            /
            len(
                negatives
            )
        ),
    }


def select_floor(
    records,
):

    candidates = []


    for threshold in (
        threshold_candidates(
            records
        )
    ):

        metrics = (
            evaluate_threshold(
                records,
                threshold,
            )
        )


        # ====================================================
        # HARD safety rule:
        #
        # This is a catastrophic-low-score floor.
        #
        # We are NOT willing to lose a manually audited
        # relevant claim during calibration.
        # ====================================================

        if (
            metrics[
                "positive_kept"
            ]
            !=
            metrics[
                "positive_total"
            ]
        ):

            continue


        candidates.append(
            metrics
        )


    if not candidates:

        raise RuntimeError(
            "No threshold preserved every positive pair."
        )


    # ========================================================
    # Among thresholds that preserve ALL positives:
    #
    # 1. reject maximum negatives
    # 2. if tied, use LOWER threshold
    #
    # Lower threshold = more conservative filtering.
    # ========================================================

    candidates.sort(
        key=lambda item: (
            -item[
                "negative_rejected"
            ],

            item[
                "threshold"
            ],
        )
    )


    return (
        candidates[
            0
        ],
        candidates,
    )


def main():

    reranker = (
        BGEReranker()
    )


    records = (
        score_pairs(
            reranker
        )
    )


    selected, candidates = (
        select_floor(
            records
        )
    )


    positives = sorted(
        (
            record
            for record
            in records
            if record[
                "label"
            ]
            ==
            1
        ),
        key=lambda item:
            item[
                "score"
            ],
    )


    negatives = sorted(
        (
            record
            for record
            in records
            if record[
                "label"
            ]
            ==
            0
        ),
        key=lambda item:
            item[
                "score"
            ],
    )


    output = {
        "selected":
            selected,

        "positive_scores":
            [
                {
                    "id":
                        item[
                            "id"
                        ],

                    "score":
                        item[
                            "score"
                        ],
                }

                for item
                in positives
            ],

        "negative_scores":
            [
                {
                    "id":
                        item[
                            "id"
                        ],

                    "score":
                        item[
                            "score"
                        ],
                }

                for item
                in negatives
            ],

        "records":
            records,

        "safe_candidates":
            candidates,
    }


    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "RELEVANCE SAFETY FLOOR CALIBRATION"
    )

    print(
        "=" * 90
    )


    print(
        "\nPOSITIVES:"
    )


    for item in positives:

        print(
            (
                f"{item['score']:>9.4f}  "
                f"{item['id']}"
            )
        )


    print(
        "\nNEGATIVES:"
    )


    for item in negatives:

        print(
            (
                f"{item['score']:>9.4f}  "
                f"{item['id']}"
            )
        )


    print(
        "\nSELECTED FLOOR:"
    )

    print(
        json.dumps(
            selected,
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
            output,
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