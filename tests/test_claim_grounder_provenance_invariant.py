from adaptive_agentic_rag.generation.claim_grounder import (
    ClaimGrounder,
)


# ============================================================
# Fake reranker
#
# TechCrunch intentionally receives a MUCH larger score.
#
# If source binding is broken, TechCrunch will win.
# ============================================================

class FakeReranker:

    def __init__(self):

        self.seen_sources = []


    def rerank(
        self,
        query,
        documents,
        top_k=5,
        batch_size=8,
    ):

        ranked = []


        for document in documents:

            item = (
                document.copy()
            )


            source = (
                document[
                    "unit"
                ][
                    "source"
                ]
            )


            self.seen_sources.append(
                source
            )


            # ------------------------------------------------
            # Deliberately make the WRONG source semantically
            # dominant.
            #
            # Hard provenance binding must prevent it from
            # even entering this ranking for source-bound
            # claims.
            # ------------------------------------------------

            if (
                source
                ==
                "TechCrunch"
            ):

                score = 100.0


            elif (
                source
                ==
                "The Age"
            ):

                score = 1.0


            else:

                score = 0.5


            item[
                "rerank_score"
            ] = (
                score
            )


            ranked.append(
                item
            )


        ranked.sort(
            key=lambda item:
                item[
                    "rerank_score"
                ],
            reverse=True,
        )


        return ranked[
            :top_k
        ]


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
            f"unit_{citation_id}",

        "citation_id":
            citation_id,

        "source":
            source,

        "title":
            f"{source} example title",

        "text":
            text,

        "provenance_text": (
            f"Source: {source}. "
            f"Title: {source} example title. "
            f"Evidence: {text}"
        ),

        "unit_type":
            "sentence",
    }


# ============================================================
# Build ClaimGrounder WITHOUT loading the real NLI model.
#
# We want an architecture/invariant test, not a model-quality
# test.
# ============================================================

def build_grounder(
    units,
):

    grounder = object.__new__(
        ClaimGrounder
    )


    grounder.reranker = (
        FakeReranker()
    )


    grounder.max_candidate_units = (
        6
    )


    # --------------------------------------------------------
    # Bypass context parsing.
    #
    # _check_claim still executes:
    #
    # source binding
    #     ↓
    # BGE candidate selection
    #     ↓
    # NLI evaluation
    #     ↓
    # ClaimSupport
    # --------------------------------------------------------

    grounder._build_evidence_units = (
        lambda context:
            units
    )


    # --------------------------------------------------------
    # Make NLI maximally permissive.
    #
    # Every candidate is considered a perfect entailment.
    #
    # Therefore the ONLY thing protecting provenance is the
    # hard structural source-binding rule.
    # --------------------------------------------------------

    grounder._predict_nli = (
        lambda premise, claim: {
            "label":
                "entailment",

            "contradiction":
                0.0005,

            "entailment":
                0.999,

            "neutral":
                0.0005,
        }
    )


    return grounder


# ============================================================
# Core invariant
# ============================================================

def test_named_source_can_never_bind_to_other_source():

    units = [
        unit(
            citation_id=1,
            source="The Age",
            text=(
                "Google faced criticism "
                "over its market practices."
            ),
        ),

        unit(
            citation_id=2,
            source="TechCrunch",
            text=(
                "Google faced an antitrust "
                "class action lawsuit."
            ),
        ),
    ]


    grounder = build_grounder(
        units
    )


    result = grounder._check_claim(
        claim=(
            "The Age reported that Google "
            "faced criticism over its "
            "market practices."
        ),

        context=None,
    )


    assert (
        result.supported
        is True
    )


    # --------------------------------------------------------
    # THE architectural invariant.
    # --------------------------------------------------------

    assert (
        result.citation_id
        ==
        1
    )


    # --------------------------------------------------------
    # The supporting premise may legitimately be either:
    #
    # plain:
    #   "Google faced criticism..."
    #
    # or provenance:
    #   "Source: The Age... Evidence: ..."
    #
    # Source identity is enforced structurally BEFORE NLI,
    # therefore supporting_text itself is not required to
    # contain the source name.
    # --------------------------------------------------------

    assert (
        result.premise_mode
        in {
            "plain",
            "provenance",
        }
    )


    assert (
        result.supporting_text
        is not None
    )


    # --------------------------------------------------------
    # THE actual architectural invariant:
    #
    # TechCrunch must never become eligible for this
    # source-bound claim.
    # --------------------------------------------------------

    assert (
        "TechCrunch"
        not in
        grounder
        .reranker
        .seen_sources
    )


    assert (
        grounder
        .reranker
        .seen_sources
        ==
        [
            "The Age"
        ]
    )

# ============================================================
# Make the adversarial condition explicit.
#
# TechCrunch gets BGE=100.
# The Age gets BGE=1.
#
# Yet TechCrunch never becomes eligible.
# ============================================================

def test_wrong_source_high_bge_score_cannot_override_binding():

    units = [
        unit(
            citation_id=10,
            source="The Age",
            text=(
                "The publication discussed "
                "Google's conduct."
            ),
        ),

        unit(
            citation_id=20,
            source="TechCrunch",
            text=(
                "A highly semantically similar "
                "Google antitrust report."
            ),
        ),
    ]


    grounder = build_grounder(
        units
    )


    result = grounder._check_claim(
        claim=(
            "The Age reported on Google's "
            "market conduct."
        ),

        context=None,
    )


    assert (
        result.citation_id
        ==
        10
    )


    assert (
        grounder
        .reranker
        .seen_sources
        ==
        [
            "The Age"
        ]
    )


# ============================================================
# Regression:
#
# A claim WITHOUT source attribution must preserve the old
# unrestricted Grounder V2 behavior.
# ============================================================

def test_claim_without_source_still_uses_global_candidates():

    units = [
        unit(
            citation_id=1,
            source="The Age",
            text=(
                "Google faced criticism."
            ),
        ),

        unit(
            citation_id=2,
            source="TechCrunch",
            text=(
                "Google faced an antitrust "
                "lawsuit."
            ),
        ),
    ]


    grounder = build_grounder(
        units
    )


    result = grounder._check_claim(
        claim=(
            "Google faced an antitrust "
            "lawsuit."
        ),

        context=None,
    )


    # --------------------------------------------------------
    # No source was explicitly named.
    #
    # Both sources must remain eligible.
    # --------------------------------------------------------

    assert set(
        grounder
        .reranker
        .seen_sources
    ) == {
        "The Age",
        "TechCrunch",
    }


    # --------------------------------------------------------
    # Fake reranker deliberately favors TechCrunch.
    #
    # This confirms unrestricted V2 behavior is preserved.
    # --------------------------------------------------------

    assert (
        result.citation_id
        ==
        2
    )


# ============================================================
# Pipe-style source invariant
# ============================================================

def test_primary_alias_of_pipe_source_binds_correctly():

    units = [
        unit(
            citation_id=1,
            source=(
                "Cnbc | "
                "World Business News Leader"
            ),
            text=(
                "Amazon sellers discussed "
                "their experience selling "
                "on the platform."
            ),
        ),

        unit(
            citation_id=2,
            source="Mashable",
            text=(
                "Amazon Cyber Monday deals "
                "continued."
            ),
        ),
    ]


    grounder = build_grounder(
        units
    )


    result = grounder._check_claim(
        claim=(
            "CNBC discussed Amazon sellers' "
            "experience selling on Amazon."
        ),

        context=None,
    )


    assert (
        result.citation_id
        ==
        1
    )


    assert (
        grounder
        .reranker
        .seen_sources
        ==
        [
            "Cnbc | "
            "World Business News Leader"
        ]
    )


# ============================================================
# Multiple explicitly named sources
#
# This verifies that binding is not accidentally reduced to
# "one source only".
# ============================================================

def test_two_named_sources_remain_eligible_but_third_is_excluded():

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


    grounder = build_grounder(
        units
    )


    result = grounder._check_claim(
        claim=(
            "The Verge and TechCrunch "
            "reported on Google's case."
        ),

        context=None,
    )


    seen = set(
        grounder
        .reranker
        .seen_sources
    )


    assert seen == {
        "The Verge",
        "TechCrunch",
    }


    assert (
        "Fortune"
        not in
        seen
    )


    assert result.citation_id in {
        1,
        2,
    }
def test_missing_named_source_cannot_fall_back_to_other_sources():

    units = [
        unit(
            citation_id=1,
            source="TechCrunch",
            text=(
                "A highly relevant Google "
                "antitrust class action."
            ),
        ),

        unit(
            citation_id=2,
            source="The Verge",
            text=(
                "Another highly relevant "
                "Google antitrust case."
            ),
        ),
    ]


    grounder = build_grounder(
        units
    )


    result = grounder._check_claim(
        claim=(
            "The Age reported a class action "
            "lawsuit against Google."
        ),

        context=None,
    )


    assert (
        result.supported
        is False
    )


    assert (
        result.citation_id
        is None
    )


    assert (
        result.supporting_text
        is None
    )


    # --------------------------------------------------------
    # Most important:
    #
    # missing explicit source must fail BEFORE BGE.
    # --------------------------------------------------------

    assert (
        grounder
        .reranker
        .seen_sources
        ==
        []
    )    