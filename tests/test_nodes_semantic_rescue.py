from types import SimpleNamespace

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)


class FakeEvidenceGrader:

    def __init__(
        self,
        *,
        sufficient,
        score,
        reasons=None,
    ):

        self.sufficient = sufficient
        self.score = score
        self.reasons = (
            reasons
            or []
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
                self.score
            ),
            reasons=list(
                self.reasons
            ),
        )


class FakeSemanticRescue:

    def __init__(
        self,
        result,
    ):

        self.result = result
        self.calls = 0


    def analyze(
        self,
        *,
        query,
        context,
        query_type,
    ):

        self.calls += 1

        return dict(
            self.result
        )


def make_nodes(
    *,
    v2_sufficient,
    rescue_sufficient,
    score=0.70,
):

    nodes = (
        RAGNodes.__new__(
            RAGNodes
        )
    )


    nodes.evidence_grader = (
        FakeEvidenceGrader(
            sufficient=(
                v2_sufficient
            ),
            score=score,
            reasons=[
                "v2 diagnostic"
            ],
        )
    )


    nodes.semantic_rescue = (
        FakeSemanticRescue(
            {
                "sufficient":
                    rescue_sufficient,

                "supported_requirement_count":
                    (
                        3
                        if rescue_sufficient
                        else 1
                    ),

                "required_requirement_count":
                    3,

                "missing_query_sources":
                    [],

                "document_diversity_required":
                    False,

                "document_diversity_ok":
                    True,

                "person_bridge_required":
                    False,

                "person_bridge_ok":
                    True,
            }
        )
    )


    return nodes


def make_state():

    return {
        "original_query":
            "Example question",

        "current_query":
            "Example question",

        "query_type":
            "multihop",

        "context":
            object(),
    }


def test_v2_accept_skips_semantic_rescue():

    nodes = make_nodes(
        v2_sufficient=True,
        rescue_sufficient=False,
        score=0.82,
    )


    result = (
        nodes.grade_evidence(
            make_state()
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
        0.82
    )


    assert (
        nodes.semantic_rescue.calls
        ==
        0
    )


    assert (
        "evidence_path=v2"
        in
        result[
            "evidence_reasons"
        ]
    )


def test_semantic_rescue_can_accept_v2_rejection():

    nodes = make_nodes(
        v2_sufficient=False,
        rescue_sufficient=True,
        score=0.71,
    )


    result = (
        nodes.grade_evidence(
            make_state()
        )
    )


    assert (
        result[
            "evidence_sufficient"
        ]
        is True
    )


    # Preserve real V2 score.
    assert (
        result[
            "evidence_score"
        ]
        ==
        0.71
    )


    assert (
        nodes.semantic_rescue.calls
        ==
        1
    )


    assert (
        "evidence_path=constrained_semantic_rescue"
        in
        result[
            "evidence_reasons"
        ]
    )


def test_failed_semantic_rescue_remains_insufficient():

    nodes = make_nodes(
        v2_sufficient=False,
        rescue_sufficient=False,
        score=0.69,
    )


    result = (
        nodes.grade_evidence(
            make_state()
        )
    )


    assert (
        result[
            "evidence_sufficient"
        ]
        is False
    )


    assert (
        nodes.semantic_rescue.calls
        ==
        1
    )


    assert (
        "semantic_rescue=insufficient"
        in
        result[
            "evidence_reasons"
        ]
    )