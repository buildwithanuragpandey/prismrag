from types import SimpleNamespace

from adaptive_agentic_rag.generation.generator import (
    GroundedGenerator,
)

from adaptive_agentic_rag.generation.relation_aware_answer_resolver import (
    RelationAwareAnswerResolver,
)


def claim(
    text: str,
    citation_id: int,
):

    return SimpleNamespace(
        claim=
            text,

        citation_id=
            citation_id,

        relevance_score=
            10.0,
    )


def make_generator():

    generator = object.__new__(
        GroundedGenerator
    )


    generator.relation_resolver = (
        RelationAwareAnswerResolver()
    )


    return generator


# ============================================================
# Canonical Case 12
#
# Qwen:
#     No
#
# Grounded + relevant facts:
#     exact same predicate
#
# Resolver:
#     Yes
# ============================================================

def test_case12_overrides_wrong_qwen_direct_answer():

    generator = (
        make_generator()
    )


    relevant_claims = [
        claim(
            (
                "The live score update and highlight excerpt "
                'for "Jaguars vs. Saints" mentioned a player '
                "achieving a first down."
            ),
            citation_id=1,
        ),

        claim(
            (
                "The live score update and highlight excerpt "
                'for "Chiefs vs. Packers" also mentioned a '
                "player achieving a first down."
            ),
            citation_id=2,
        ),
    ]


    (
        direct_answer,
        resolution,
    ) = (
        generator
        ._resolve_direct_answer(
            query=(
                "Has the reporting style regarding live "
                "score updates and highlights remained "
                "consistent between the two articles?"
            ),

            draft_direct_answer=
                "No",

            relevant_claims=
                relevant_claims,
        )
    )


    assert (
        resolution.applied
        is True
    )


    assert (
        direct_answer
        ==
        "Yes"
    )


# ============================================================
# Case 14:
#
# Baseline No is already correct.
#
# Different predicates mean resolver must stay out.
# ============================================================

def test_case14_preserves_existing_direct_answer():

    generator = (
        make_generator()
    )


    relevant_claims = [
        claim(
            (
                "Taylor Swift revealed her connection with "
                "Travis Kelce in July after he confessed on "
                "his podcast."
            ),
            citation_id=1,
        ),

        claim(
            (
                "Swift confirmed that she attended Kelce's "
                "game at Arrowhead Stadium in September, "
                "dating him at the time."
            ),
            citation_id=1,
        ),
    ]


    (
        direct_answer,
        resolution,
    ) = (
        generator
        ._resolve_direct_answer(
            query=(
                "Was the news about Taylor Swift's "
                "relationship inconsistent with the "
                "later report?"
            ),

            draft_direct_answer=
                "No",

            relevant_claims=
                relevant_claims,
        )
    )


    assert (
        resolution.applied
        is False
    )


    assert (
        direct_answer
        ==
        "No"
    )


# ============================================================
# Non-relation question:
#
# Entity answer must never be touched.
# ============================================================

def test_entity_answer_is_never_overridden():

    generator = (
        make_generator()
    )


    relevant_claims = [
        claim(
            "Sam Bankman-Fried founded FTX.",
            citation_id=1,
        ),

        claim(
            "Sam Bankman-Fried faced fraud charges.",
            citation_id=2,
        ),
    ]


    (
        direct_answer,
        resolution,
    ) = (
        generator
        ._resolve_direct_answer(
            query=(
                "Who is the individual described "
                "in both reports?"
            ),

            draft_direct_answer=
                "Sam Bankman-Fried",

            relevant_claims=
                relevant_claims,
        )
    )


    assert (
        resolution.applied
        is False
    )


    assert (
        direct_answer
        ==
        "Sam Bankman-Fried"
    )


# ============================================================
# Single supported/relevant fact:
#
# cross-report consistency cannot be resolved.
# ============================================================

def test_single_relevant_claim_never_overrides():

    generator = (
        make_generator()
    )


    relevant_claims = [
        claim(
            (
                "The report mentioned a player "
                "achieving a first down."
            ),
            citation_id=1,
        ),
    ]


    (
        direct_answer,
        resolution,
    ) = (
        generator
        ._resolve_direct_answer(
            query=(
                "Did the reports remain consistent?"
            ),

            draft_direct_answer=
                "No",

            relevant_claims=
                relevant_claims,
        )
    )


    assert (
        resolution.applied
        is False
    )


    assert (
        direct_answer
        ==
        "No"
    )


# ============================================================
# Final answer construction must use RESOLVED answer while
# keeping citations from verified facts.
# ============================================================

def test_resolved_answer_builds_with_verified_citations():

    generator = (
        make_generator()
    )


    relevant_claims = [
        claim(
            (
                "The live score update and highlight excerpt "
                'for "Jaguars vs. Saints" mentioned a player '
                "achieving a first down."
            ),
            citation_id=1,
        ),

        claim(
            (
                "The live score update and highlight excerpt "
                'for "Chiefs vs. Packers" also mentioned a '
                "player achieving a first down."
            ),
            citation_id=2,
        ),
    ]


    (
        direct_answer,
        resolution,
    ) = (
        generator
        ._resolve_direct_answer(
            query=(
                "Did the reports remain consistent?"
            ),

            draft_direct_answer=
                "No",

            relevant_claims=
                relevant_claims,
        )
    )


    assert (
        resolution.applied
        is True
    )


    final_answer = (
        generator
        ._build_grounded_answer(
            direct_answer=
                direct_answer,

            relevant_claims=
                relevant_claims,
        )
    )


    assert (
        final_answer.startswith(
            "Yes [1][2]"
        )
    )


    assert (
        "[1]"
        in
        final_answer
    )


    assert (
        "[2]"
        in
        final_answer
    )