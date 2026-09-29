import re


# ============================================================
# Sentence-boundary exceptions
#
# These abbreviations should not normally terminate a sentence
# when followed by the remainder of the same expression.
# ============================================================

_SENTENCE_SPLIT_PATTERN = re.compile(

    r"(?<!\bv\.)"
    r"(?<!\bvs\.)"
    r"(?<!\bMr\.)"
    r"(?<!\bMs\.)"
    r"(?<!\bDr\.)"
    r"(?<!\bProf\.)"
    r"(?<!\bJr\.)"
    r"(?<!\bSr\.)"

    r"(?<=[.!?])"
    r"\s+",

    flags=re.IGNORECASE,
)


def split_sentences(
    text: str,
    *,
    split_newlines: bool = False,
    min_chars: int = 1,
) -> list[str]:

    text = (
        text
        or ""
    )


    text = (
        text
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .strip()
    )


    if not text:

        return []


    # ========================================================
    # First preserve explicit line boundaries when requested.
    #
    # ContextBuilder deals with bullet-heavy source documents,
    # so newlines are useful boundaries there.
    #
    # ClaimGrounder normally works on already compacted text
    # and does not need to force every newline into a sentence.
    # ========================================================

    if split_newlines:

        blocks = [

            block.strip()

            for block
            in re.split(
                r"\n+",
                text,
            )

            if block.strip()
        ]

    else:

        blocks = [
            text
        ]


    sentences = []


    for block in blocks:

        block = re.sub(
            r"\s+",
            " ",
            block,
        ).strip()


        if not block:

            continue


        parts = (
            _SENTENCE_SPLIT_PATTERN.split(
                block
            )
        )


        for part in parts:

            part = (
                part
                .strip()
            )


            if (
                part
                and
                len(part) >= min_chars
            ):

                sentences.append(
                    part
                )


    return sentences