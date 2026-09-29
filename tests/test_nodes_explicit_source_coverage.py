from types import SimpleNamespace

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)

from adaptive_agentic_rag.orchestration.explicit_source_coverage import (
    ExplicitSourceCoverageGuard,
)


# ============================================================
# Test helpers
# ============================================================

def context(
    *sources,
):

    return SimpleNamespace(
        items=[
            SimpleNamespace(
                source=source,
            )

            for source
            in sources
        ],

        text="test context",
    )


class FakeEvidenceGrader:

    def __init__(
        self,
        *,
        sufficient=True,
        evidence_score=0.91,
    ):

        self.sufficient = (
            sufficient
        )


        self.evidence_score = (
            evidence_score
        )


    def grade(
        self,
        *,
        query,
        context,
        query_type,
    ):

        return SimpleNamespace(
            sufficient=(
                self.sufficient
            ),

            evidence_score=(
                self.evidence_score
            ),

            reasons=[
                "Fake V2 grade."
            ],
        )


class FailIfCalledSemanticRescue:
    """
    Missing explicit source must reject BEFORE
    semantic rescue.
    """

    def analyze(
        self,
        **kwargs,
    ):

        raise AssertionError(
            "Semantic rescue must not run "
            "when explicit source coverage fails."
        )


class FakeSemanticRescue:

    def __init__(
        self,
        *,
        sufficient,
    ):

        self.sufficient = (
            sufficient
        )


    def analyze(
        self,
        *,
        query,
        context,
        query_type,
    ):

        return SimpleNamespace(
            sufficient=(
                self.sufficient
            ),

            threshold=-6.127460241317749,

            required_fraction=0.75,

            requirement_count=2,

            supported_requirement_count=2,

            required_requirement_count=2,

            supporting_document_ids=[
                "doc_a",
                "doc_b",
            ],
        )


# ============================================================
# Case 15 invariant
#
# Even if V2 says:
#
#     sufficient=True
#     score=0.91
#
# missing The Age must override that decision.
# ============================================================

def test_missing_explicit_source_overrides_high_v2_score():

    nodes = object.__new__(
        RAGNodes
    )


    nodes.evidence_grader = (
        FakeEvidenceGrader(
            sufficient=True,
            evidence_score=0.91,
        )
    )


    nodes.source_coverage_guard = (
        ExplicitSourceCoverageGuard()
    )


    nodes.semantic_rescue = (
        FailIfCalledSemanticRescue()
    )


    state = {
        "original_query": (
            "Has Google's portrayal in reports by "
            "The Age remained consistent with "
            "The Verge's coverage and "
            "TechCrunch's report?"
        ),

        "query_type":
            "complex",

        "context":
            context(
                "The Verge",
                "TechCrunch",
            ),
    }


    result = (
        nodes.grade_evidence(
            state
        )
    )


    assert (
        result[
            "evidence_sufficient"
        ]
        is False
    )


    # --------------------------------------------------------
    # Preserve actual V2 numeric score.
    # --------------------------------------------------------

    assert (
        result[
            "evidence_score"
        ]
        ==
        0.91
    )


    reasons = " ".join(
        result[
            "evidence_reasons"
        ]
    ).lower()


    assert (
        "explicit source coverage failed"
        in
        reasons
    )


    assert (
        "age"
        in
        reasons
    )


    assert (
        "explicit_source_coverage_reject"
        in
        reasons
    )


# ============================================================
# All explicit sources present:
#
# V2 sufficient must retain the fast path.
# Semantic rescue must NOT run.
# ============================================================

def test_all_sources_present_preserves_v2_fast_path():

    nodes = object.__new__(
        RAGNodes
    )


    nodes.evidence_grader = (
        FakeEvidenceGrader(
            sufficient=True,
            evidence_score=0.86,
        )
    )


    nodes.source_coverage_guard = (
        ExplicitSourceCoverageGuard()
    )


    nodes.semantic_rescue = (
        FailIfCalledSemanticRescue()
    )


    state = {
        "original_query": (
            "Has Google's portrayal in reports by "
            "The Age remained consistent with "
            "The Verge's coverage and "
            "TechCrunch's report?"
        ),

        "query_type":
            "complex",

        "context":
            context(
                "The Age",
                "The Verge",
                "TechCrunch",
            ),
    }


    result = (
        nodes.grade_evidence(
            state
        )
    )


    assert (
        result[
            "evidence_sufficient"
        ]
        is True
    )


    assert (
        result[
            "evidence_score"
        ]
        ==
        0.86
    )


    reasons = " ".join(
        result[
            "evidence_reasons"
        ]
    ).lower()


    assert (
        "evidence_path=v2"
        in
        reasons
    )


    assert (
        "explicit_source_coverage=satisfied"
        in
        reasons
    )


# ============================================================
# Source coverage passes but V2 fails:
#
# semantic rescue must remain available.
# ============================================================

def test_source_guard_does_not_disable_semantic_rescue():

    nodes = object.__new__(
        RAGNodes
    )


    nodes.evidence_grader = (
        FakeEvidenceGrader(
            sufficient=False,
            evidence_score=0.72,
        )
    )


    nodes.source_coverage_guard = (
        ExplicitSourceCoverageGuard()
    )


    nodes.semantic_rescue = (
        FakeSemanticRescue(
            sufficient=True,
        )
    )


    state = {
        "original_query": (
            "According to TechCrunch, "
            "which company released the product?"
        ),

        "query_type":
            "multihop",

        "context":
            context(
                "TechCrunch",
                "The Verge",
            ),
    }


    result = (
        nodes.grade_evidence(
            state
        )
    )


    assert (
        result[
            "evidence_sufficient"
        ]
        is True
    )


    # --------------------------------------------------------
    # Semantic rescue may change boolean sufficiency,
    # but it must NOT fake-promote the V2 score.
    # --------------------------------------------------------

    assert (
        result[
            "evidence_score"
        ]
        ==
        0.72
    )


    reasons = " ".join(
        result[
            "evidence_reasons"
        ]
    ).lower()


    assert (
        "semantic_rescue_attempted=true"
        in
        reasons
    )


    assert (
        "semantic_rescue_sufficient=true"
        in
        reasons
    )


    assert (
        "semantic_rescue=sufficient"
        in
        reasons
    )


    assert (
        "evidence_path=constrained_semantic_rescue"
        in
        reasons
    )


# ============================================================
# Source coverage passes + V2 fails + rescue fails:
#
# final decision remains insufficient.
# ============================================================

def test_failed_semantic_rescue_remains_insufficient():

    nodes = object.__new__(
        RAGNodes
    )


    nodes.evidence_grader = (
        FakeEvidenceGrader(
            sufficient=False,
            evidence_score=0.69,
        )
    )


    nodes.source_coverage_guard = (
        ExplicitSourceCoverageGuard()
    )


    nodes.semantic_rescue = (
        FakeSemanticRescue(
            sufficient=False,
        )
    )


    state = {
        "original_query": (
            "According to TechCrunch, "
            "which company released the product?"
        ),

        "query_type":
            "multihop",

        "context":
            context(
                "TechCrunch",
            ),
    }


    result = (
        nodes.grade_evidence(
            state
        )
    )


    assert (
        result[
            "evidence_sufficient"
        ]
        is False
    )


    assert (
        result[
            "evidence_score"
        ]
        ==
        0.69
    )


    reasons = " ".join(
        result[
            "evidence_reasons"
        ]
    ).lower()


    assert (
        "semantic_rescue=insufficient"
        in
        reasons
    )


    assert (
        "evidence_path=constrained_semantic_rescue_reject"
        in
        reasons
    )

# ============================================================
# Query without explicit publishers:
#
# Guard must be transparent.
# ============================================================

def test_no_explicit_source_does_not_create_false_rejection():

    nodes = object.__new__(
        RAGNodes
    )


    nodes.evidence_grader = (
        FakeEvidenceGrader(
            sufficient=True,
            evidence_score=0.84,
        )
    )


    nodes.source_coverage_guard = (
        ExplicitSourceCoverageGuard()
    )


    nodes.semantic_rescue = (
        FailIfCalledSemanticRescue()
    )


    state = {
        "original_query": (
            "Which company faced an "
            "antitrust lawsuit?"
        ),

        "query_type":
            "simple",

        "context":
            context(
                "TechCrunch",
                "The Verge",
            ),
    }


    result = (
        nodes.grade_evidence(
            state
        )
    )


    assert (
        result[
            "evidence_sufficient"
        ]
        is True
    )


    assert (
        result[
            "evidence_score"
        ]
        ==
        0.84
    )