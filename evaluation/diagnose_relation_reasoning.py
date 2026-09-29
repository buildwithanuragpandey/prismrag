import re

from dataclasses import dataclass


STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "for",
    "from",
    "to",
    "in",
    "on",
    "at",
    "by",
    "with",
    "and",
    "or",
    "also",
    "was",
    "were",
    "is",
    "are",
    "has",
    "have",
    "had",
    "did",
    "does",
    "do",
    "that",
    "this",
    "their",
    "its",
}


@dataclass
class RelationDiagnostic:
    relation_type: str | None

    requested_polarity: str | None

    fact_count: int

    normalized_facts: list[list[str]]

    pairwise_overlap: list[float]

    minimum_overlap: float | None

    maximum_overlap: float | None


def tokenize(
    text: str,
) -> list[str]:

    # Remove quoted entity/event names.
    #
    # Example:
    #
    # "Jaguars vs. Saints"
    #
    # We want the remaining predicate:
    #
    # live score update ... mentioned player first down

    text = re.sub(
        r'"[^"]+"',
        " ",
        text,
    )


    text = re.sub(
        r"'[^']+'",
        " ",
        text,
    )


    text = text.lower()


    tokens = re.findall(
        r"[a-z0-9]+",
        text,
    )


    output = []


    for token in tokens:

        if token in STOPWORDS:

            continue


        # Dates/numbers are usually entity-specific noise
        # for this diagnostic.

        if token.isdigit():

            continue


        output.append(
            token
        )


    return output


def jaccard(
    left: list[str],
    right: list[str],
) -> float:

    left_set = set(
        left
    )


    right_set = set(
        right
    )


    if not left_set or not right_set:

        return 0.0


    intersection = (
        left_set
        &
        right_set
    )


    union = (
        left_set
        |
        right_set
    )


    return (
        len(
            intersection
        )
        /
        len(
            union
        )
    )


def detect_relation(
    query: str,
) -> tuple[str | None, str | None]:

    normalized = (
        query.lower()
    )


    # ========================================================
    # Consistency-positive
    # ========================================================

    if any(
        phrase in normalized

        for phrase in [
            "remained consistent",
            "remain consistent",
            "consistent perspective",
            "common trend",
            "show a consistent",
        ]
    ):

        return (
            "consistency",
            "positive",
        )


    # ========================================================
    # Explicit inconsistency question
    #
    # Example:
    #
    # "Was X inconsistent with Y?"
    #
    # A consistent pair implies answer NO.
    # ========================================================

    if (
        "inconsistent"
        in normalized
    ):

        return (
            "consistency",
            "negative",
        )


    return (
        None,
        None,
    )


def analyze(
    query: str,
    facts: list[str],
) -> RelationDiagnostic:

    relation_type, polarity = (
        detect_relation(
            query
        )
    )


    normalized_facts = [
        tokenize(
            fact
        )

        for fact
        in facts
    ]


    overlaps = []


    for left_index in range(
        len(
            normalized_facts
        )
    ):

        for right_index in range(
            left_index + 1,
            len(
                normalized_facts
            ),
        ):

            overlaps.append(
                jaccard(
                    normalized_facts[
                        left_index
                    ],

                    normalized_facts[
                        right_index
                    ],
                )
            )


    return RelationDiagnostic(
        relation_type=
            relation_type,

        requested_polarity=
            polarity,

        fact_count=
            len(
                facts
            ),

        normalized_facts=
            normalized_facts,

        pairwise_overlap=
            overlaps,

        minimum_overlap=(
            min(
                overlaps
            )
            if overlaps
            else None
        ),

        maximum_overlap=(
            max(
                overlaps
            )
            if overlaps
            else None
        ),
    )


def print_case(
    *,
    name: str,
    query: str,
    facts: list[str],
):

    print(
        "\n"
        +
        "=" * 100
    )

    print(
        name
    )

    print(
        "=" * 100
    )


    result = analyze(
        query=
            query,

        facts=
            facts,
    )


    print(
        "Relation type:",
        result.relation_type,
    )


    print(
        "Requested polarity:",
        result.requested_polarity,
    )


    print(
        "Fact count:",
        result.fact_count,
    )


    print(
        "\nNormalized facts:"
    )


    for index, tokens in enumerate(
        result.normalized_facts,
        start=1,
    ):

        print(
            f"{index}:",
            tokens,
        )


    print(
        "\nPairwise overlap:",
        [
            round(
                score,
                4,
            )

            for score
            in result.pairwise_overlap
        ],
    )


    print(
        "Minimum overlap:",
        (
            round(
                result.minimum_overlap,
                4,
            )

            if (
                result.minimum_overlap
                is not None
            )

            else None
        ),
    )


    print(
        "Maximum overlap:",
        (
            round(
                result.maximum_overlap,
                4,
            )

            if (
                result.maximum_overlap
                is not None
            )

            else None
        ),
    )


def main():

    # ========================================================
    # CASE 12
    #
    # Current answer is wrong despite both grounded facts
    # being supported.
    # ========================================================

    print_case(
        name=
            "CASE 12 — Sporting News consistency",

        query=(
            "Has the reporting style regarding live score "
            "updates and highlights from NFL games by "
            "Sporting News remained consistent between the "
            'article featuring "Jaguars vs. Saints" and '
            'the one covering "Chiefs vs. Packers", '
            "considering the excerpts mentioning a player "
            "achieving a first down?"
        ),

        facts=[
            (
                'The live score update and highlight excerpt '
                'for "Jaguars vs. Saints" mentioned a player '
                "achieving a first down."
            ),

            (
                'The live score update and highlight excerpt '
                'for "Chiefs vs. Packers" also mentioned a '
                "player achieving a first down."
            ),
        ],
    )


    # ========================================================
    # CASE 14
    #
    # Current DIRECT_ANSWER = No is already correct.
    #
    # We do NOT want a generic consistency resolver to
    # blindly overwrite this.
    # ========================================================

    print_case(
        name=
            "CASE 14 — Taylor / Kelce inconsistency",

        query=(
            "Was the news about Taylor Swift's relationship "
            "with Travis Kelce inconsistent with the later "
            "report from The Independent - Life and Style?"
        ),

        facts=[
            (
                "Taylor Swift revealed her connection with "
                "Travis Kelce in July after he confessed on "
                "his podcast."
            ),

            (
                "Swift confirmed that she attended Kelce's "
                "game at Arrowhead Stadium in September, "
                "dating him at the time."
            ),
        ],
    )


    # ========================================================
    # CONTROL
    #
    # Two unrelated facts should have low overlap and must
    # not produce a strong deterministic consistency signal.
    # ========================================================

    print_case(
        name=
            "CONTROL — unrelated evidence",

        query=(
            "Did the two reports remain consistent?"
        ),

        facts=[
            (
                "Amazon shares fell after an antitrust "
                "lawsuit was filed."
            ),

            (
                "Artists are seeking record deals with more "
                "control and better economics."
            ),
        ],
    )


if __name__ == "__main__":

    main()