from adaptive_agentic_rag.generation.relation_aware_answer_resolver import (
    RelationAwareAnswerResolver,
)


def resolver():

    return (
        RelationAwareAnswerResolver()
    )


# ============================================================
# CASE 12
#
# Canonical bug:
#
# grounded facts agree structurally,
# DIRECT_ANSWER was wrong.
# ============================================================

def test_case12_exact_consistency_resolves_yes():

    result = resolver().resolve(
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
                "The live score update and highlight excerpt "
                'for "Jaguars vs. Saints" mentioned a player '
                "achieving a first down."
            ),

            (
                "The live score update and highlight excerpt "
                'for "Chiefs vs. Packers" also mentioned a '
                "player achieving a first down."
            ),
        ],
    )


    assert (
        result.applied
        is True
    )


    assert (
        result.resolved_answer
        ==
        "Yes"
    )


    assert (
        result.relation_type
        ==
        "consistency"
    )


    assert (
        result.requested_polarity
        ==
        "positive"
    )


    assert (
        result.predicate_signatures[
            0
        ]
        ==
        result.predicate_signatures[
            1
        ]
    )


# ============================================================
# Negative wording:
#
# If the same predicates are explicitly asked whether they
# are inconsistent, answer must be No.
# ============================================================

def test_exact_equivalence_resolves_inconsistency_question_no():

    result = resolver().resolve(
        query=(
            "Were the two reports inconsistent with "
            "each other?"
        ),

        facts=[
            (
                'The report for "Event A" mentioned a '
                "player achieving a first down."
            ),

            (
                'The report for "Event B" mentioned a '
                "player achieving a first down."
            ),
        ],
    )


    assert (
        result.applied
        is True
    )


    assert (
        result.resolved_answer
        ==
        "No"
    )


# ============================================================
# CASE 14
#
# Correct baseline answer was No.
#
# Facts are related but NOT structurally equivalent.
#
# Resolver must stay out.
# ============================================================

def test_case14_does_not_override():

    result = resolver().resolve(
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


    assert (
        result.applied
        is False
    )


    assert (
        result.resolved_answer
        is None
    )


# ============================================================
# CONTROL
#
# Unrelated grounded facts must never produce Yes.
# ============================================================

def test_unrelated_facts_do_not_resolve():

    result = resolver().resolve(
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


    assert (
        result.applied
        is False
    )


# ============================================================
# Case 10-style facts
#
# Multiple valid facts about completely different predicates.
#
# Do not turn them into a synthetic consistency answer.
# ============================================================

def test_case10_style_different_predicates_do_not_resolve():

    result = resolver().resolve(
        query=(
            "Do the articles show a consistent perspective?"
        ),

        facts=[
            (
                "The Mashable article suggests that Amazon's "
                "Cyber Monday includes continued and new "
                "deals."
            ),

            (
                "The Sydney Morning Herald article focuses "
                "on an antitrust lawsuit affecting Amazon's "
                "stock price."
            ),
        ],
    )


    assert (
        result.applied
        is False
    )


# ============================================================
# Case 9-style facts
#
# Same topic does not mean same predicate.
# ============================================================

def test_same_topic_different_predicates_do_not_resolve():

    result = resolver().resolve(
        query=(
            "Do the reports show a consistent perspective?"
        ),

        facts=[
            (
                "Gary Wang pleaded guilty to federal "
                "criminal charges."
            ),

            (
                "Caroline Ellison pleaded guilty to federal "
                "criminal charges."
            ),
        ],
    )


    # --------------------------------------------------------
    # They are semantically parallel, but the entities are
    # not quoted and therefore the signatures differ.
    #
    # Conservative resolver stays out.
    # --------------------------------------------------------

    assert (
        result.applied
        is False
    )


# ============================================================
# No relation
# ============================================================

def test_non_consistency_question_is_ignored():

    result = resolver().resolve(
        query=(
            "Who won the game?"
        ),

        facts=[
            (
                'The report for "Game A" said the team won.'
            ),

            (
                'The report for "Game B" said the team won.'
            ),
        ],
    )


    assert (
        result.applied
        is False
    )


    assert (
        result.relation_type
        is None
    )


# ============================================================
# One fact is insufficient for cross-report consistency.
# ============================================================

def test_single_fact_is_not_enough():

    result = resolver().resolve(
        query=(
            "Did the reports remain consistent?"
        ),

        facts=[
            (
                "The report mentioned a player achieving "
                "a first down."
            ),
        ],
    )


    assert (
        result.applied
        is False
    )


# ============================================================
# Very short signatures are unsafe.
#
# Even if they happen to match exactly.
# ============================================================

def test_short_matching_predicates_are_not_enough():

    result = resolver().resolve(
        query=(
            "Did the reports remain consistent?"
        ),

        facts=[
            '"Team A" won.',
            '"Team B" won.',
        ],
    )


    assert (
        result.applied
        is False
    )


# ============================================================
# Negation mismatch
#
# Must never be treated as equivalent.
# ============================================================

def test_negation_mismatch_is_not_equivalent():

    result = resolver().resolve(
        query=(
            "Did the reports remain consistent?"
        ),

        facts=[
            (
                'The report for "Event A" mentioned a '
                "player achieving a first down."
            ),

            (
                'The report for "Event B" did not mention '
                "a player achieving a first down."
            ),
        ],
    )


    assert (
        result.applied
        is False
    )