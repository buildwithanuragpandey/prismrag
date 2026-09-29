from adaptive_agentic_rag.generation.atomic_claim_extractor import (
    AtomicClaimExtractor,
)


def extractor():

    return AtomicClaimExtractor()


def test_does_not_split_legal_v_abbreviation():

    result = extractor().extract(
        (
            "The Verge reports that Google engaged in "
            "anticompetitive practices during the "
            "Epic v. Google trial."
        )
    )


    assert result.claims == [
        (
            "The Verge reports that Google engaged in "
            "anticompetitive practices during the "
            "Epic v. Google trial."
        )
    ]


def test_does_not_split_vs_abbreviation():

    result = extractor().extract(
        (
            "Swift attended the Chiefs vs. Chargers game."
        )
    )


    assert result.claims == [
        (
            "Swift attended the Chiefs vs. Chargers game."
        )
    ]


def test_does_not_split_compound_subject():

    result = extractor().extract(
        (
            "Taylor Swift and Travis Kelce "
            "have been dating for several weeks."
        )
    )


    assert result.claims == [
        (
            "Taylor Swift and Travis Kelce "
            "have been dating for several weeks."
        )
    ]


def test_does_not_split_subordinate_compound_subject():

    result = extractor().extract(
        (
            "The independent reports indicate that "
            "Swift and Kelce have been dating "
            "for several weeks."
        )
    )


    assert result.claims == [
        (
            "The independent reports indicate that "
            "Swift and Kelce have been dating "
            "for several weeks."
        )
    ]


def test_splits_two_real_independent_clauses():

    result = extractor().extract(
        (
            "Amazon hosts Black Friday sales and "
            "Walmart starts its promotion earlier."
        )
    )


    assert result.claims == [
        "Amazon hosts Black Friday sales.",
        "Walmart starts its promotion earlier.",
    ]


def test_does_not_split_us_abbreviation():

    result = extractor().extract(
        (
            "The U.S. government filed an antitrust case."
        )
    )


    assert result.claims == [
        (
            "The U.S. government filed an antitrust case."
        )
    ]