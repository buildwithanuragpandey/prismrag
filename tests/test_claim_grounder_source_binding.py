from adaptive_agentic_rag.generation.claim_grounder import (
    ClaimGrounder,
)


# ============================================================
# Evidence-unit helper
# ============================================================

def unit(
    *,
    citation_id,
    source,
    text,
):

    return {
        "id":
            f"evidence_{citation_id}",

        "citation_id":
            citation_id,

        "source":
            source,

        "title":
            "Example",

        "text":
            text,

        "provenance_text": (
            f"Source: {source}. "
            f"Title: Example. "
            f"Evidence: {text}"
        ),

        "unit_type":
            "sentence",
    }


# ============================================================
# Existing-source binding
# ============================================================

def test_claim_explicitly_naming_source_filters_other_sources():

    units = [
        unit(
            citation_id=1,
            source="The Age",
            text="The Age evidence.",
        ),

        unit(
            citation_id=2,
            source="TechCrunch",
            text="TechCrunch evidence.",
        ),
    ]


    result = (
        ClaimGrounder._apply_source_binding(
            claim=(
                "The Age reported that "
                "Google faced criticism."
            ),

            units=
                units,
        )
    )


    assert len(
        result
    ) == 1


    assert (
        result[
            0
        ][
            "source"
        ]
        ==
        "The Age"
    )


# ============================================================
# No source attribution:
# preserve original global behavior
# ============================================================

def test_claim_without_source_preserves_all_units():

    units = [
        unit(
            citation_id=1,
            source="The Age",
            text="First evidence.",
        ),

        unit(
            citation_id=2,
            source="TechCrunch",
            text="Second evidence.",
        ),
    ]


    result = (
        ClaimGrounder._apply_source_binding(
            claim=(
                "Google faced an "
                "antitrust lawsuit."
            ),

            units=
                units,
        )
    )


    assert len(
        result
    ) == 2


# ============================================================
# Pipe-style source alias
# ============================================================

def test_pipe_source_primary_alias_is_detected():

    units = [
        unit(
            citation_id=1,
            source=(
                "Cnbc | "
                "World Business News Leader"
            ),
            text=(
                "Amazon sellers discussed "
                "selling on Amazon."
            ),
        ),

        unit(
            citation_id=2,
            source="Mashable",
            text="Cyber Monday deals.",
        ),
    ]


    result = (
        ClaimGrounder._apply_source_binding(
            claim=(
                "The CNBC article discusses "
                "selling on Amazon."
            ),

            units=
                units,
        )
    )


    assert len(
        result
    ) == 1


    assert (
        result[
            0
        ][
            "citation_id"
        ]
        ==
        1
    )


# ============================================================
# Alias safety:
# "The Age" must never become generic "age"
# ============================================================

def test_the_age_is_not_reduced_to_generic_age():

    aliases = (
        ClaimGrounder._source_aliases(
            "The Age"
        )
    )


    assert (
        "the age"
        in
        aliases
    )


    assert (
        "age"
        not in
        aliases
    )


# ============================================================
# Safe removal of leading "The" for multi-word sources
# ============================================================

def test_multiword_the_source_gets_short_alias():

    aliases = (
        ClaimGrounder._source_aliases(
            "The Sydney Morning Herald"
        )
    )


    assert (
        "the sydney morning herald"
        in
        aliases
    )


    assert (
        "sydney morning herald"
        in
        aliases
    )


# ============================================================
# Multiple explicitly named sources
# ============================================================

def test_claim_can_bind_to_two_explicit_sources():

    units = [
        unit(
            citation_id=1,
            source="The Verge",
            text="Verge evidence.",
        ),

        unit(
            citation_id=2,
            source="TechCrunch",
            text="TechCrunch evidence.",
        ),

        unit(
            citation_id=3,
            source="Fortune",
            text="Fortune evidence.",
        ),
    ]


    result = (
        ClaimGrounder._apply_source_binding(
            claim=(
                "The Verge and TechCrunch "
                "reported different aspects "
                "of the Google case."
            ),

            units=
                units,
        )
    )


    selected_sources = {
        item[
            "source"
        ]

        for item
        in result
    }


    assert selected_sources == {
        "The Verge",
        "TechCrunch",
    }


# ============================================================
# NEW:
# Explicitly attributed source is missing from context.
#
# IMPORTANT:
#
# This must FAIL CLOSED.
#
# Claim:
#
#   "The Age reported..."
#
# Context:
#
#   TechCrunch
#   The Verge
#
# The system must NOT fall back to either available source.
# ============================================================

def test_missing_explicit_source_fails_closed():

    units = [
        unit(
            citation_id=1,
            source="TechCrunch",
            text=(
                "Google faced a class action "
                "antitrust lawsuit."
            ),
        ),

        unit(
            citation_id=2,
            source="The Verge",
            text=(
                "Google faced another "
                "antitrust dispute."
            ),
        ),
    ]


    result = (
        ClaimGrounder._apply_source_binding(
            claim=(
                "The Age reported a class action "
                "lawsuit against Google."
            ),

            units=
                units,
        )
    )


    assert (
        result
        ==
        []
    )


# ============================================================
# NEW:
# "The Fortune article ..." is also explicit provenance.
#
# Fortune absent from context must not fall back to
# TechCrunch.
# ============================================================

def test_missing_source_article_attribution_fails_closed():

    units = [
        unit(
            citation_id=1,
            source="TechCrunch",
            text="FTX evidence.",
        ),
    ]


    result = (
        ClaimGrounder._apply_source_binding(
            claim=(
                "The Fortune article states "
                "that Sam Bankman-Fried was "
                "responsible for the fraud."
            ),

            units=
                units,
        )
    )


    assert (
        result
        ==
        []
    )


# ============================================================
# NEW regression:
#
# A normal factual claim with NO source attribution must not
# accidentally become fail-closed.
#
# Original Grounder V2 behavior must remain intact.
# ============================================================

def test_non_provenance_claim_still_preserves_global_units():

    units = [
        unit(
            citation_id=1,
            source="TechCrunch",
            text="Google evidence.",
        ),

        unit(
            citation_id=2,
            source="The Verge",
            text="More Google evidence.",
        ),
    ]


    result = (
        ClaimGrounder._apply_source_binding(
            claim=(
                "Google faced antitrust "
                "scrutiny."
            ),

            units=
                units,
        )
    )


    assert len(
        result
    ) == 2